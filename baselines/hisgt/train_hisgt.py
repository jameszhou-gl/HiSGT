import pandas as pd
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import pickle
import time
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from HiSGT import HiSGT
from embedding_utils import (
    create_label_special_token_dict,
    load_semantic_embeddings,
    load_hierarchy_embeddings,
    create_label_special_token_semantic_embeddings,
    prepare_total_vocab_semantic_embeddings,
    validate_vocab_embeddings
)
import pdb
# pdb.set_trace()
import warnings
warnings.filterwarnings('ignore')

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Argument parsing
parser = argparse.ArgumentParser(
    description='Train the HiSGT model')
parser.add_argument('--dataset', type=str,
                    default='mimiciii', help="Dataset name")
parser.add_argument('--dataset_version', type=str,
                    default='1.4_convert_icd10', help="Dataset version")
parser.add_argument('--total_vocab_size', type=int, default=7014)
parser.add_argument('--code_vocab_size', type=int, default=6984)
parser.add_argument('--label_vocab_size', type=int, default=25)
parser.add_argument('--special_vocab_size', type=int, default=5)
parser.add_argument('--n_positions', type=int, default=750)
parser.add_argument('--n_ctx', type=int, default=700)
parser.add_argument('--n_embd', type=int, default=384)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--layer_norm_epsilon', type=float, default=1e-5)
parser.add_argument('--initializer_range', type=float, default=0.02)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--LOG_DIR', type=str, required=True)
parser.add_argument('--alpha', type=float, default=1.0, help="Weight for sequence loss")
parser.add_argument('--beta', type=float, default=0.0, help="Weight for semantic mse loss")
parser.add_argument('--gamma', type=float, default=0.0, help="Weight for hierarchy mse loss")
parser.add_argument('--flag_vec_semantic', action="store_true", help="whether to use semantic embeddings")
parser.add_argument('--flag_vec_hierarchy', action="store_true",
                    help="whether to use hierarchy embeddings")
parser.add_argument('--wandb_log', action="store_false", help="Log to wandb")

args = parser.parse_args()
print(args)

if args.beta>0.0:
    assert args.flag_vec_semantic == True, "Semantic embeddings are required for calculating semantic loss"
if args.gamma>0.0:
    assert args.flag_vec_hierarchy == True, "Hierarchy embeddings are required for calculating hierarchy loss"

if args.wandb_log:
    import wandb
    wandb.init(project="hisgt", name=f'{args.dataset}_{args.dataset_version}_alpha_{args.alpha}_beta_{args.beta}_gamma_{args.gamma}', config=args)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

label_special_tokens_dict = create_label_special_token_dict(args.dataset, args.dataset_version, args.code_vocab_size, args.label_vocab_size)
# Load ICD and special token embeddings
code_semantic_embeddings, icd_codes = load_semantic_embeddings(
    args.dataset, args.dataset_version, args.code_vocab_size)
label_special_semantic_embeddings, label_special_token_embeddings_dict = create_label_special_token_semantic_embeddings(
    label_special_tokens_dict)
assert label_special_semantic_embeddings.shape[0] == args.special_vocab_size+args.label_vocab_size, f"Special token embeddings size mismatch! Expected {args.special_vocab_size+args.label_vocab_size}, got {label_special_semantic_embeddings.shape[0]}"
# Prepare vocab embeddings
total_vocab_semantic_embeddings, semantic_dim = prepare_total_vocab_semantic_embeddings(
    code_semantic_embeddings, label_special_semantic_embeddings, args.total_vocab_size, device=device)
# Validate vocab embeddings
validate_vocab_embeddings(total_vocab_semantic_embeddings, icd_codes,
                          label_special_tokens_dict, code_semantic_embeddings, label_special_semantic_embeddings)
total_vocab_semantic_embeddings_path = os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_total_vocab_semantic_embeddings.npy")
np.save(total_vocab_semantic_embeddings_path, total_vocab_semantic_embeddings.cpu().numpy())
print(f"Total vocab semantic embeddings saved to {total_vocab_semantic_embeddings_path}")
total_hierarchy_embeddings, hierarchy_dim = load_hierarchy_embeddings(
    args.dataset, args.dataset_version, args.code_vocab_size, args.total_vocab_size, device=device)
if total_hierarchy_embeddings is not None:
    assert hierarchy_dim == args.n_embd, f"Hierarchy embeddings dimension is expected to match n_emdb, otherwise the model needs extra projection layer as the semantic embeddings"
    total_vocab_hierarchy_embeddings_path = os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_total_vocab_hierarchy_embeddings.npy")
    np.save(total_vocab_hierarchy_embeddings_path,
            total_hierarchy_embeddings.cpu().numpy())
    print(f"Total vocab embeddings saved to {total_vocab_hierarchy_embeddings_path}")
else:
    # print("No hierarchy embeddings found.")
    assert not args.flag_vec_hierarchy, "Hierarchy embeddings not found, but flag_vec_hierarchy is set to True"

# Load datasets
train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
orig_train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))
val_data_path = f"data/{args.dataset}/{args.dataset_version}/val.pkl"
orig_val_ehr_dataset = pickle.load(open(val_data_path, 'rb'))


def prepare_ehr_dataset(original_dataset, label_special_tokens_dict, n_ctx):
    """Prepare tokenized EHR dataset using special tokens."""
    ehr_dataset = []
    for orig_ehr in original_dataset:
        # Initialize with PADDING
        new_ehr = [label_special_tokens_dict["PADDING"]["index"]] * n_ctx
        idx = 0
        new_ehr[idx] = label_special_tokens_dict["START_RECORD"]["index"]
        idx += 1

        # Add Labels
        for l in orig_ehr['labels'].nonzero()[0]:
            new_ehr[idx] = label_special_tokens_dict[f"LABEL_{l}"]["index"]
            idx += 1
        new_ehr[idx] = label_special_tokens_dict["END_LABEL"]["index"]
        idx += 1

        # Add Visits
        for v in orig_ehr['visits']:
            for c in v:
                new_ehr[idx] = c
                idx += 1
            new_ehr[idx] = label_special_tokens_dict["END_VISIT"]["index"]
            idx += 1

        # End Record
        new_ehr[idx] = label_special_tokens_dict["END_RECORD"]["index"]
        ehr_dataset.append(new_ehr)
    return ehr_dataset


# Prepare training and validation datasets
train_ehr_dataset = prepare_ehr_dataset(
    orig_train_ehr_dataset, label_special_tokens_dict, args.n_ctx)
val_ehr_dataset = prepare_ehr_dataset(
    orig_val_ehr_dataset, label_special_tokens_dict, args.n_ctx)


def get_batch(loc, batch_size, mode):
    if mode == 'train':
        ehr = train_ehr_dataset[loc:loc + batch_size]
    elif mode == 'valid':
        ehr = val_ehr_dataset[loc:loc + batch_size]

    batch_ehr = np.array(ehr)
    batch_ehr_tensor = torch.tensor(batch_ehr, dtype=torch.long, device=device)
    batch_semantic_tensor = total_vocab_semantic_embeddings[batch_ehr_tensor]
    batch_hierarchy_tensor = total_hierarchy_embeddings[batch_ehr_tensor] if total_hierarchy_embeddings is not None else None

    return batch_ehr_tensor, batch_semantic_tensor, batch_hierarchy_tensor


def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)


# Initialize the model and optimizer
model = HiSGT(args, semantic_dim, hierarchy_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training loop
train_start_time = time.time()
global_loss = 1e10
patience = 10
no_improvement_nums = 0

for e in tqdm(range(args.epoch)):
    epoch_start_time = time.time()
    shuffle_training_data(train_ehr_dataset)

    for i in range(0, len(train_ehr_dataset), args.batch_size):
        model.train()
        # batch_ehr, batch_semantic = get_batch(i, args.batch_size, 'train')
        batch_ehr_tensor, batch_semantic_tensor, batch_hierarchy_tensor = get_batch(
            i, args.batch_size, 'train')

        optimizer.zero_grad()
        # Forward pass
        sequence_loss, semantic_loss, hierarchy_loss, predicted_semantics, _ = model(
            batch_ehr_tensor,
            position_ids=None,
            ehr_labels=batch_ehr_tensor,
            semantic_embeddings=batch_semantic_tensor,
            hierarchy_embeddings=batch_hierarchy_tensor
        )
        # Combine losses
        total_loss = args.alpha * sequence_loss
        if args.beta > 0.0:
            # ! even if beta is 0.0, semantic loss is calculated and involved in gradient computation, so we use a conditional statement here
            total_loss += args.beta * semantic_loss
        if args.gamma > 0.0:
            total_loss += args.gamma * hierarchy_loss
        total_loss.backward()
        optimizer.step()
        
        if i % (100 * args.batch_size) == 0:
            print(f"Epoch {e}, Iter {i}: Training Loss: {total_loss:.6f}, Sequence Loss: {sequence_loss:.6f}, Semantic Loss: {semantic_loss:.6f}, Hierarchy Loss: {hierarchy_loss:.6f}")
    
        if i % (250 * args.batch_size) == 0:
            # Validation loop after each epoch
            if i == 0:
                continue
            model.eval()
            with torch.no_grad():
                val_losses = []
                for v_i in range(0, len(val_ehr_dataset), args.batch_size):
                    batch_ehr_tensor, batch_semantic_tensor, batch_hierarchy_tensor = get_batch(
                        v_i, args.batch_size, 'valid')

                    val_sequence_loss, val_semantic_loss, val_hierarchy_loss, predicted_semantics, _ = model(
                        batch_ehr_tensor,
                        position_ids=None,
                        ehr_labels=batch_ehr_tensor,
                        semantic_embeddings=batch_semantic_tensor,
                        hierarchy_embeddings=batch_hierarchy_tensor
                    )
                    val_total_loss = args.alpha * val_sequence_loss
                    if args.beta > 0.0:
                        # ! even if beta is 0.0, semantic loss is calculated and involved in gradient computation, so we use a conditional statement here
                        val_total_loss += args.beta * val_semantic_loss
                    if args.gamma > 0.0:
                        # ! even if beta is 0.0, semantic loss is calculated and involved in gradient computation, so we use a conditional statement here
                        val_total_loss += args.gamma * val_hierarchy_loss
                    val_losses.append(val_total_loss.cpu().item())
               
                cur_val_loss = np.mean(val_losses)
                print(f"Epoch {e} Validation Loss: {cur_val_loss:.6f}")
                # if args.wandb_log:
                print({
                    "epoch": e, 
                    "iteration": i,
                    "training_loss": total_loss.item(),
                    "train_sequence_loss": sequence_loss.item(),
                    "train_semantic_loss": semantic_loss.item(),
                    "train_hierarchy_loss": hierarchy_loss.item(),
                    "val_loss": cur_val_loss.item(),
                    "val_sequence_loss": val_sequence_loss.item(),
                    "val_semantic_loss": val_semantic_loss.item(),
                    "val_hierarchy_loss": val_hierarchy_loss.item(),
                    })

                if cur_val_loss < global_loss:
                    print(f"Validation loss improved from {global_loss:.6f} to {cur_val_loss:.6f}. Saving model...")
                    global_loss = cur_val_loss
                    no_improvement_nums = 0

                    # Save the best model
                    train_end_time = time.time()
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iteration': e,
                        'best_val_loss': global_loss
                    }
                    torch.save(state, os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_hisgt.pt"))

                else:
                    no_improvement_nums += 1
                    print(f"No improvement for {no_improvement_nums} times.")

    if no_improvement_nums >= patience:
        print(f"Early stopping triggered after {e + 1} epochs.")
        break

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {e} Time: {epoch_time:.2f} seconds")

# Save training metrics
total_training_time = time.time() - train_start_time
computational_metrics = {
    "training_time_seconds": total_training_time,
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "batch_size": args.batch_size,
    "epochs": e + 1,
    "learning_rate": args.lr
}
with open(os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_computational_cost.json"), "w") as f:
    json.dump(computational_metrics, f, indent=4)

print(f"Training completed in {total_training_time:.2f} seconds")
