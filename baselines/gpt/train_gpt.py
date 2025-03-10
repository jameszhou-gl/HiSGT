import os
import torch
import numpy as np
import random
import argparse
import pickle
import time
import json
from tqdm import tqdm
from gpt import GPTModel
# from config import GPTConfig

import warnings
warnings.filterwarnings('ignore')

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# config = GPTConfig()

# Argument parsing
parser = argparse.ArgumentParser(description='Train the gpt model')
parser.add_argument('--dataset', type=str,
                    default='mimiciii', help="Dataset name")
parser.add_argument('--dataset_version', type=str,
                    default='1.4', help="Dataset version")
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
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--LOG_DIR', type=str, required=True)
args = parser.parse_args()
# Print parsed arguments
print(args)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
orig_train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))
val_data_path = f"data/{args.dataset}/{args.dataset_version}/val.pkl"
orig_val_ehr_dataset = pickle.load(open(val_data_path, 'rb'))


train_ehr_dataset = []
for orig_ehr in orig_train_ehr_dataset:
    new_ehr = [args.total_vocab_size - 1] * args.n_ctx  # Pad Codes
    new_ehr[0] = args.code_vocab_size + \
        args.label_vocab_size  # Start Record
    idx = 1

    # Add Labels
    for l in orig_ehr['labels'].nonzero()[0]:
        new_ehr[idx] = l + args.code_vocab_size
        idx += 1

    new_ehr[idx] = args.code_vocab_size + \
        args.label_vocab_size + 1  # End Labels
    idx += 1

    # Add Visits
    for v in orig_ehr['visits']:
        for c in v:
            new_ehr[idx] = c
            idx += 1
        new_ehr[idx] = args.code_vocab_size + \
            args.label_vocab_size + 2  # End Visit
        idx += 1

    new_ehr[idx] = args.code_vocab_size + \
        args.label_vocab_size + 3  # End Record
    train_ehr_dataset.append(new_ehr)

val_ehr_dataset = []
for orig_ehr in orig_val_ehr_dataset:
    new_ehr = [args.total_vocab_size - 1] * args.n_ctx  # Pad Codes
    new_ehr[0] = args.code_vocab_size + \
        args.label_vocab_size  # Start Record
    idx = 1

    # Add Labels
    for l in orig_ehr['labels'].nonzero()[0]:
        new_ehr[idx] = l + args.code_vocab_size
        idx += 1

    new_ehr[idx] = args.code_vocab_size + \
        args.label_vocab_size + 1  # End Labels
    idx += 1

    # Add Visits
    for v in orig_ehr['visits']:
        for c in v:
            new_ehr[idx] = c
            idx += 1
        new_ehr[idx] = args.code_vocab_size + \
            args.label_vocab_size + 2  # End Visit
        idx += 1

    new_ehr[idx] = args.code_vocab_size + \
        args.label_vocab_size + 3  # End Record
    val_ehr_dataset.append(new_ehr)


def get_batch(loc, batch_size, mode):
    if mode == 'train':
        ehr = train_ehr_dataset[loc:loc+batch_size]
    elif mode == 'valid':
        ehr = val_ehr_dataset[loc:loc+batch_size]
    else:
        ehr = test_ehr_dataset[loc:loc+batch_size]

    batch_ehr = np.array(ehr)
    return batch_ehr


def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)


model = GPTModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# if os.path.exists("../../save/gpt_model"):
#   print("Loading previous model")
#   checkpoint = torch.load('../../save/gpt_model', map_location=torch.device(device))
#   model.load_state_dict(checkpoint['model'])
#   optimizer.load_state_dict(checkpoint['optimizer'])
#   model.set_tied()

# Train
train_start_time = time.time()
global_loss = 1e10
for e in tqdm(range(args.epoch)):
    epoch_start_time = time.time()
    shuffle_training_data(train_ehr_dataset)
    for i in range(0, len(train_ehr_dataset), args.batch_size):
        model.train()

        batch_ehr = get_batch(i, args.batch_size, 'train')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.long).to(device)

        optimizer.zero_grad()
        loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr)
        loss.backward()
        optimizer.step()
        free, total = torch.cuda.mem_get_info(device)
        gpu_memory = (total - free)/ (1024 ** 2)  # In MB
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if i % (100*args.batch_size) == 0:
            print("Epoch %d, Iter %d: Training Loss:%.6f" % (e, i, loss))
        if i % (250*args.batch_size) == 0:
            if i == 0:
                continue

            model.eval()
            with torch.no_grad():
                val_l = []
                for v_i in range(0, len(val_ehr_dataset), args.batch_size):
                    batch_ehr = get_batch(v_i, args.batch_size, 'valid')
                    batch_ehr = torch.tensor(
                        batch_ehr, dtype=torch.long).to(device)

                    val_loss, _, _ = model(
                        batch_ehr, position_ids=None, ehr_labels=batch_ehr)
                    val_l.append((val_loss).cpu().detach().numpy())

                cur_val_loss = np.mean(val_l)
                print("Epoch %d Validation Loss:%.7f" % (e, cur_val_loss))
                if cur_val_loss < global_loss:
                    train_end_time = time.time()
                    global_loss = cur_val_loss
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iteration': i
                    }
                    torch.save(state, os.path.join(args.LOG_DIR, f'{args.dataset}_{args.dataset_version}_gpt.pt'))
                    print("Best model saved.")
    
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {e} Time: {epoch_time:.2f} seconds")

# Post-training metrics
total_training_time = train_end_time - train_start_time
# Save metrics to JSON
computational_metrics = {
    "training_time_seconds": total_training_time,
    "training_gpu_memory_mb": gpu_memory,
    "total_parameters": num_params,
    "trainable_parameters": trainable_params,
    "batch_size": args.batch_size,
    "epochs": args.epoch,
    "learning_rate": args.lr
}

with open(os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_computational_cost.json"), "w") as f:
    json.dump(computational_metrics, f, indent=4)

print(f"Training Time: {total_training_time:.2f} seconds")
print(f"Training GPU Memory Used: {gpu_memory:.2f} MB")
print(f"Total Parameters: {num_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Saved computational metrics to {args.LOG_DIR}")
