import os
import torch
import pickle
import argparse
import random
import json
import time 
import numpy as np
from tqdm import tqdm
from HiSGT import HiSGT
import torch.nn.functional as F
from embedding_utils import create_label_special_token_dict
import pdb
# pdb.set_trace()
import psutil
process = psutil.Process(os.getpid())

import warnings
warnings.filterwarnings('ignore')

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Argument parsing
parser = argparse.ArgumentParser(
    description='Sample sequences from the trained HiSGT model')
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
parser.add_argument('--alpha', type=float, default=1.0,
                    help="Weight for sequence loss")
parser.add_argument('--beta', type=float, default=0.0, help="Weight for semantic mse loss")
parser.add_argument('--gamma', type=float, default=0.0, help="Weight for hierarchy mse loss")
parser.add_argument('--flag_vec_semantic', action="store_true",
                    help="whether to use semantic embeddings")
parser.add_argument('--flag_vec_hierarchy', action="store_true",
                    help="whether to use hierarchy embeddings")
args = parser.parse_args()
# Print parsed arguments
print(args)

# Load or initialize computational cost metrics
computational_cost_path = os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_computational_cost.json")
if os.path.exists(computational_cost_path):
    with open(computational_cost_path, "r") as f:
        computational_cost = json.load(f)
else:
    computational_cost = {}

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

label_special_tokens_dict = create_label_special_token_dict(args.dataset, args.dataset_version, args.code_vocab_size, args.label_vocab_size)


# Load semantic embeddings
total_vocab_semantic_embeddings_path = os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_total_vocab_semantic_embeddings.npy")
total_vocab_semantic_embeddings = torch.tensor(
    np.load(total_vocab_semantic_embeddings_path), dtype=torch.float32).to(device)
print(f"Loaded total vocab semantic embeddings from {total_vocab_semantic_embeddings_path} with size: {total_vocab_semantic_embeddings.size()}")
semantic_dim = total_vocab_semantic_embeddings.size(-1)
# Load hierarchy embeddings
total_vocab_hierarchy_embeddings_path = os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_total_vocab_hierarchy_embeddings.npy")
if os.path.exists(total_vocab_hierarchy_embeddings_path):
    total_vocab_hierarchy_embeddings = torch.tensor(
        np.load(total_vocab_hierarchy_embeddings_path), dtype=torch.float32).to(device)
    print(f"Loaded total vocab hierarchy embeddings from {total_vocab_hierarchy_embeddings_path} with size: {total_vocab_hierarchy_embeddings.size()}")
    hierarchy_dim = total_vocab_hierarchy_embeddings.size(-1)
else:
    total_vocab_hierarchy_embeddings = None
    hierarchy_dim = args.n_embd
    
train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))


# model = HiSGT(args, semantic_dim=768).to(device)
model = HiSGT(args, semantic_dim, hierarchy_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

checkpoint = torch.load(
    f'{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_hisgt.pt', map_location=device)
print(f'load model from hisgt.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])


def sample_sequence(model, length, context, total_vocab_semantic_embeddings, total_vocab_hierarchy_embeddings, batch_size=None, device='cuda', sample=True):
    """
    Generate synthetic sequences using the model, incorporating semantic embeddings.
    """
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(
        0).repeat(batch_size, 1)
    prev = context
    ehr = context
    past = None

    with torch.no_grad():
        for _ in range(length):
            # Fetch semantic embeddings for the current context
            # Lookup embeddings for the tokens in `prev`
            semantic_embeddings = total_vocab_semantic_embeddings[prev]
            hierarchy_embeddings = total_vocab_hierarchy_embeddings[prev] if total_vocab_hierarchy_embeddings is not None else None

            # Forward pass through the model with semantic embeddings
            code_logits, past = model(
                prev, semantic_embeddings=semantic_embeddings, hierarchy_embeddings=hierarchy_embeddings, past=past)
            code_logits = code_logits[:, -1, :]
            log_probs = F.softmax(code_logits, dim=-1)

            # Sampling or greedy decoding
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                prev = torch.argmax(log_probs, dim=1)

            ehr = torch.cat((ehr, prev), dim=1)

            # Early stopping if END_RECORD token is generated
            if all([args.code_vocab_size + args.label_vocab_size + 3 in ehr[i] for i in range(batch_size)]):
                print(f"Early stopping triggered at sequence length {_ + 1}")
                break

    ehr = ehr.cpu().detach().numpy()
    return ehr



def convert_ehr(ehrs, index_to_code=None):
    ehr_outputs = []
    for i in range(len(ehrs)):
        ehr = ehrs[i]
        ehr_output = []
        visit_output = []
        labels_output = np.zeros(args.label_vocab_size)
        started_visits = False
        for j in range(1, len(ehr)):
            code = ehr[j]
            if not started_visits:
                # if code == args.code_vocab_size + args.label_vocab_size + 1:
                if code == label_special_tokens_dict['END_LABEL']['index']:
                    started_visits = True
                elif code >= args.code_vocab_size and code < args.code_vocab_size + args.label_vocab_size:
                    labels_output[code - args.code_vocab_size] = 1

            else:
                if code < args.code_vocab_size:
                    if code not in visit_output:
                        visit_output.append(
                            index_to_code[code] if index_to_code is not None else code)
                elif code == label_special_tokens_dict['END_VISIT']['index']:
                    if visit_output != []:
                        ehr_output.append(visit_output)
                        visit_output = []
                elif code == label_special_tokens_dict['END_RECORD']['index']:
                    break

        if visit_output != []:
            ehr_output.append(visit_output)

        if index_to_code is not None:
            labels_output = [index_to_code[idx + args.code_vocab_size]
                             for idx in np.nonzero(labels_output)[0]]

        ehr_outputs.append({'visits': ehr_output, 'labels': labels_output})
    ehr = None
    ehr_output = None
    labels_output = None
    visit_output = None
    return ehr_outputs

# Start timer and memory usage tracking
start_time = time.time()
start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
stoken = [args.code_vocab_size+args.label_vocab_size]
for i in tqdm(range(0, len(train_ehr_dataset), 2*args.batch_size), desc="Generating synthetic EHRs"):
    bs = min([len(train_ehr_dataset)-i, 2*args.batch_size])
    batch_synthetic_ehrs = sample_sequence(
        model, args.n_ctx, stoken, total_vocab_semantic_embeddings, total_vocab_hierarchy_embeddings, batch_size=bs, device=device, sample=True)
    batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
    synthetic_ehr_dataset += batch_synthetic_ehrs
    # Monitor system memory
    mem_info = process.memory_info()
    print(f"Iteration {i}: System Memory Usage: {mem_info.rss / 1e6:.2f} MB")
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        print(f"Iteration {i}: Allocated Memory: {allocated_memory / 1e6} MB, Reserved Memory: {reserved_memory / 1e6} MB")


# End timer and memory usage tracking
sampling_time = time.time() - start_time
free, total = torch.cuda.mem_get_info(device)
gpu_memory = (total - free)/ (1024 ** 2)  # In MB
# Update computational cost metrics
computational_cost["total_sampling_time_seconds"] = sampling_time
computational_cost["per_sampling_time_seconds"] = sampling_time/len(synthetic_ehr_dataset)
computational_cost["sampling_gpu_memory_mb"] = gpu_memory

# Save updated computational cost metrics
with open(computational_cost_path, "w") as f:
    json.dump(computational_cost, f, indent=4)

print(f"Sampling computational cost metrics saved to {computational_cost_path}")

pickle.dump(synthetic_ehr_dataset, open(
    f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_hisgt.pkl", 'wb'))
