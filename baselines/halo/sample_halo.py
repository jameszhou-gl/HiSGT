import torch
import json
import os
import time 
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from HALO import HALOModel
import pdb
import psutil

process = psutil.Process(os.getpid())


# Set a fixed seed for reproducibility
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Argument parsing
parser = argparse.ArgumentParser(
    description='Sample sequences from the trained HALO model')
parser.add_argument('--dataset', type=str,
                    default='mimiciii', help="Dataset name")
parser.add_argument('--dataset_version', type=str,
                    default='1.4', help="Dataset version")
parser.add_argument('--total_vocab_size', type=int,
                    default=7012, help="Total vocabulary size")
parser.add_argument('--code_vocab_size', type=int,
                    default=6984, help="Code vocabulary size")
parser.add_argument('--label_vocab_size', type=int,
                    default=25, help="Label vocabulary size")
parser.add_argument('--special_vocab_size', type=int,
                    default=3, help="Special vocabulary size")
parser.add_argument('--n_positions', type=int, default=56,
                    help="")
parser.add_argument('--n_ctx', type=int, default=48,
                    help="Context window size")
parser.add_argument('--n_embd', type=int, default=768,
                    help="Embedding dimension")
parser.add_argument('--n_layer', type=int, default=12,
                    help="Number of layers in the model")
parser.add_argument('--n_head', type=int, default=12,
                    help="")
parser.add_argument('--layer_norm_epsilon', type=float,
                    default=1e-5, help="")
parser.add_argument('--initializer_range', type=float,
                    default=0.02, help="")
parser.add_argument('--pos_loss_weight', type=float,
                    default=None, help="")
parser.add_argument('--batch_size', type=int, default=48,
                    help="Batch size for training")
parser.add_argument('--train_epochs', type=int, default=50,
                    help="Number of training epochs")
parser.add_argument('--learning_rate', type=float,
                    default=1e-4, help="Learning rate")
parser.add_argument('--sample_batch_size', type=int,
                    default=256)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--LOG_DIR', type=str, required=True)


args = parser.parse_args()
# Print parsed arguments
print(args)
if args.debug:
    pdb.set_trace()
    
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

train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))
test_data_path = f"data/{args.dataset}/{args.dataset_version}/test.pkl"
# test_ehr_dataset = pickle.load(open(test_data_path, 'rb'))
# attempt to derive vocab_size from the dataset
with open(f"data/{args.dataset}/{args.dataset_version}/meta.pkl", 'rb') as f:
    meta = pickle.load(f)
index_to_code = meta['itos']
# index_to_code_path = f"data/{args.dataset}/{args.dataset_version}/indexToCode.json"
# # Load index-to-code mapping from JSON and convert to a dictionary
# with open(index_to_code_path, 'r') as f:
#     index_to_code = json.load(f)
train_c = set([c for p in train_ehr_dataset for v in p['visits'] for c in v])
# ! filtering out any codes that the model has not seen during training
# test_ehr_dataset = [{'labels': p['labels'], 'visits': [
#     [c for c in v if c in train_c] for v in p['visits']]} for p in test_ehr_dataset]

model = HALOModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

checkpoint = torch.load(
    f'{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_halo.pt', map_location=device)
print(f'load model from halo.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
    # [256, 1, 7012]
    empty = torch.zeros((1, 1, args.total_vocab_size),
                        device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
    # [256, 7012]
    context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(
        0).repeat(batch_size, 1)
    # [256, 1, 7012]
    prev = context.unsqueeze(1)
    context = None
    with torch.no_grad():
        for _ in range(length-1):
            prev = model.sample(torch.cat((prev, empty), dim=1), sample)
            # todo check the break condition
            if torch.sum(torch.sum(prev[:, :, args.code_vocab_size+args.label_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
                break
    ehr = prev.cpu().detach().numpy()
    prev = None
    empty = None
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
start_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0


# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
stoken = np.zeros(args.total_vocab_size)
stoken[args.code_vocab_size+args.label_vocab_size] = 1
time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
for i in tqdm(range(0, len(train_ehr_dataset), args.sample_batch_size)):
    bs = min([len(train_ehr_dataset)-i, args.sample_batch_size])
    # debug
    if args.debug:
        print('debug mode: setting batch size to 8')
        bs = 8
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
    if i!=0 and i % 50 == 0:
        pickle.dump(synthetic_ehr_dataset, open(f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_halo_{time_stamp}_count_{i}.pkl", 'wb'))

pickle.dump(synthetic_ehr_dataset, open(f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_halo.pkl", 'wb'))

# End timer and memory usage tracking
sampling_time = time.time() - start_time
free, total = torch.cuda.mem_get_info(device)
gpu_memory = (total - free) / (1024 ** 2)  # In MB
# Update computational cost metrics
computational_cost["total_sampling_time_seconds"] = sampling_time
computational_cost["per_sampling_time_seconds"] = sampling_time / \
    len(synthetic_ehr_dataset)
computational_cost["sampling_gpu_memory_mb"] = gpu_memory

# Save updated computational cost metrics
with open(computational_cost_path, "w") as f:
    json.dump(computational_cost, f, indent=4)

print(f"Sampling computational cost metrics saved to {computational_cost_path}")

# pickle.dump(synthetic_ehr_dataset, open(
#     f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_gpt.pkl", 'wb'))
