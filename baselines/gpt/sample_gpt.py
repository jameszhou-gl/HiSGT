import os
import torch
import pickle
import argparse
import random
import json
import time 
import numpy as np
from tqdm import tqdm
from gpt import GPTModel
import torch.nn.functional as F
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
    description='Sample sequences from the trained gpt model')
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
with open(f"data/{args.dataset}/{args.dataset_version}/meta.pkl", 'rb') as f:
    meta = pickle.load(f)
index_to_code = meta['itos']
# index_to_code = pickle.load(open("../../data/mimiciii/1.4/indexToCode.pkl", "rb"))

# Add the labels to the index_to_code mapping
index_to_code[args.code_vocab_size] = "Chronic Condition: Alzheimer or related disorders or senile"
index_to_code[args.code_vocab_size+1] = "Chronic Condition: Heart Failure"
index_to_code[args.code_vocab_size +
              2] = "Chronic Condition: Chronic Kidney Disease"
index_to_code[args.code_vocab_size+3] = "Chronic Condition: Cancer"
index_to_code[args.code_vocab_size +
              4] = "Chronic Condition: Chronic Obstructive Pulmonary Disease"
index_to_code[args.code_vocab_size+5] = "Chronic Condition: Depression"
index_to_code[args.code_vocab_size+6] = "Chronic Condition: Diabetes"
index_to_code[args.code_vocab_size +
              7] = "Chronic Condition: Ischemic Heart Disease"
index_to_code[args.code_vocab_size+8] = "Chronic Condition: Osteoporosis"
index_to_code[args.code_vocab_size +
              9] = "Chronic Condition: rheumatoid arthritis and osteoarthritis (RA/OA)"
index_to_code[args.code_vocab_size +
              10] = "Chronic Condition: Stroke/transient Ischemic Attack"

model = GPTModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

checkpoint = torch.load(
    f'{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_gpt.pt', map_location=device)
print(f'load model from gpt.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])


def sample_sequence(model, length, context, batch_size=None, device='cuda', sample=True):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(
        0).repeat(batch_size, 1)
    prev = context
    ehr = context
    past = None
    with torch.no_grad():
        for _ in range(length):
            code_logits, past = model(prev, past=past)
            code_logits = code_logits[:, -1, :]
            log_probs = F.softmax(code_logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                prev = torch.argmax(log_probs, dim=1)
            ehr = torch.cat((ehr, prev), dim=1)

            if all([args.code_vocab_size + args.label_vocab_size + 3 in ehr[i] for i in range(batch_size)]):  # early stopping
                break
    ehr = ehr.cpu().detach().numpy()
    next = None
    prev = None
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
                if code == args.code_vocab_size + args.label_vocab_size + 1:
                    started_visits = True
                elif code >= args.code_vocab_size and code < args.code_vocab_size + args.label_vocab_size:
                    labels_output[code - args.code_vocab_size] = 1

            else:
                if code < args.code_vocab_size:
                    if code not in visit_output:
                        visit_output.append(
                            index_to_code[code] if index_to_code is not None else code)
                elif code == args.code_vocab_size + args.label_vocab_size + 2:
                    if visit_output != []:
                        ehr_output.append(visit_output)
                        visit_output = []
                elif code == args.code_vocab_size + args.label_vocab_size + 3:
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
for i in tqdm(range(0, len(train_ehr_dataset), 2*args.batch_size)):
    bs = min([len(train_ehr_dataset)-i, 2*args.batch_size])
    batch_synthetic_ehrs = sample_sequence(
        model, args.n_ctx, stoken, batch_size=bs, device=device, sample=True)
    batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
    synthetic_ehr_dataset += batch_synthetic_ehrs

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
    f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_gpt.pkl", 'wb'))
