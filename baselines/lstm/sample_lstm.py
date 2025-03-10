import os
import time
import torch
import pickle
import json
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from LSTM import LSTMBaseline
import argparse
import psutil

process = psutil.Process(os.getpid())
# Argument parsing
parser = argparse.ArgumentParser(
    description='Sample sequences from the trained LSTM model')
parser.add_argument('--dataset', type=str,
                    default='mimiciii', help="Dataset name")
parser.add_argument('--dataset_version', type=str,
                    default='1.4', help="Dataset version")
parser.add_argument('--LOG_DIR', type=str, required=True)
parser.add_argument("--total_vocab_size", type=int,
                    default=7012, help="Total vocabulary size")
parser.add_argument("--code_vocab_size", type=int,
                    default=6984, help="Code vocabulary size")
parser.add_argument("--label_vocab_size", type=int,
                    default=25, help="Label vocabulary size")
parser.add_argument("--special_vocab_size", type=int,
                    default=3, help="Special vocabulary size")
parser.add_argument("--n_positions", type=int,
                    default=56, help="Number of positions")
parser.add_argument("--n_ctx", type=int, default=48,
                    help="Context window size")
parser.add_argument("--n_embd", type=int, default=768,
                    help="Embedding size")
parser.add_argument("--n_layer", type=int, default=12,
                    help="Number of layers")
parser.add_argument("--n_head", type=int, default=12,
                    help="Number of attention heads")
parser.add_argument("--layer_norm_epsilon", type=float,
                    default=1e-5, help="Layer norm epsilon")
parser.add_argument("--initializer_range", type=float,
                    default=0.02, help="Initializer range")
parser.add_argument("--batch_size", type=int,
                    default=128, help="Batch size for training")
parser.add_argument("--epoch", type=int, default=25,
                    help="Number of epochs")
parser.add_argument("--pos_loss_weight", type=float,
                    default=None, help="Positive loss weight")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

# Load or initialize computational cost metrics
computational_cost_path = os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_computational_cost.json")
if os.path.exists(computational_cost_path):
    with open(computational_cost_path, "r") as f:
        computational_cost = json.load(f)
else:
    computational_cost = {}
    
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


local_rank = -1
fp16 = False
if local_rank == -1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))
# test_data_path = f"data/{args.dataset}/{args.dataset_version}/test.pkl"
# test_ehr_dataset = pickle.load(open(test_data_path, 'rb'))
with open(f"data/{args.dataset}/{args.dataset_version}/meta.pkl", 'rb') as f:
    meta = pickle.load(f)
index_to_code = meta['itos']

index_to_label_path = f"data/{args.dataset}/{args.dataset_version}/idToLabel.json"
with open(index_to_label_path, 'r') as f:
    id_to_label = json.load(f)
# Add the labels to the index_to_code mapping
for ind, label in id_to_label.items():
# for k, l in enumerate(id_to_label):
    index_to_code[args.code_vocab_size+int(ind)] = f"Chronic Condition: {label}"


model = LSTMBaseline(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

checkpoint = torch.load(f'{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_lstm.pt', map_location=device)

model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])


def sample_sequence(model, length, context, batch_size=None, device='cuda', sample=True):
    context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(
        0).repeat(batch_size, 1)
    prev = context.unsqueeze(1)
    with torch.no_grad():
        for i in range(length-1):
            code_probs = model(prev)
            code_probs = code_probs[:, -1, :].unsqueeze(1)
            if sample:
                visit = torch.bernoulli(code_probs)
            else:
                visit = torch.round(code_probs)
            prev = torch.cat((prev, visit), dim=1)
    ehr = prev.cpu().detach().numpy()
    visit = None
    prev = None
    return ehr


def convert_ehr(ehrs, index_to_code=None):
    ehr_outputs = []
    for i in range(len(ehrs)):
        ehr = ehrs[i]
        ehr_output = []
        labels_output = ehr[1][args.code_vocab_size:
                               args.code_vocab_size+args.label_vocab_size]
        if index_to_code is not None:
            labels_output = [index_to_code[idx + args.code_vocab_size]
                             for idx in np.nonzero(labels_output)[0]]
        for j in range(2, len(ehr)):
            visit = ehr[j]
            visit_output = []
            indices = np.nonzero(visit)[0]
            end = False
            for idx in indices:
                if idx < args.code_vocab_size:
                    visit_output.append(
                        index_to_code[idx] if index_to_code is not None else idx)
                elif idx == args.code_vocab_size+args.label_vocab_size+1:
                    end = True
            if visit_output != []:
                ehr_output.append(visit_output)
            if end:
                break
        ehr_outputs.append({'visits': ehr_output, 'labels': labels_output})
    ehr = None
    ehr_output = None
    labels_output = None
    visit = None
    visit_output = None
    indices = None
    return ehr_outputs


# Start timer and memory usage tracking
start_time = time.time()
start_memory = torch.cuda.memory_allocated(
    device) if torch.cuda.is_available() else 0


# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
stoken = np.zeros(args.total_vocab_size)
stoken[args.code_vocab_size+args.label_vocab_size] = 1
for i in tqdm(range(0, len(train_ehr_dataset), args.batch_size)):
    bs = min([len(train_ehr_dataset)-i, args.batch_size])
    batch_synthetic_ehrs = sample_sequence(
        model, args.n_ctx, stoken, batch_size=bs, device=device, sample=True)
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
    f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_lstm.pkl", 'wb'))
