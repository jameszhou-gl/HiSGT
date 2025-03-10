import os
import torch
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from synteg import Generator, DependencyModel


SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Argument parsing
parser = argparse.ArgumentParser(
    description='Synteg')
parser.add_argument('--dataset', type=str,
                    default='mimiciii', help="Dataset name")
parser.add_argument('--dataset_version', type=str,
                    default='1.4', help="Dataset version")
parser.add_argument('--LOG_DIR', type=str, required=True)
parser.add_argument("--embedding_dim", type=int,
                    default=112, help="Embedding dimension")
parser.add_argument("--word_embedding_dim", type=int,
                    default=80, help="Word embedding dimension")
parser.add_argument("--attention_size", type=int,
                    default=128, help="Attention size")
parser.add_argument("--ff_dim", type=int, default=128,
                    help="Feed-forward dimension")
parser.add_argument("--max_num_visit", type=int, default=48,
                    help="Maximum number of visits per patient")
parser.add_argument("--max_length_visit", type=int,
                    default=80, help="Maximum number of codes per visit")
parser.add_argument("--num_head", type=int, default=4,
                    help="Number of attention heads")
parser.add_argument("--code_vocab_dim", type=int,
                    default=6984, help="Code vocabulary dimension")
parser.add_argument("--label_vocab_dim", type=int,
                    default=25, help="Label vocabulary dimension")
parser.add_argument("--vocab_dim", type=int, default=6984 + 25 + 2,
                    help="Total vocabulary dimension (codes + labels + special tokens)")
parser.add_argument("--head_dim", type=int, default=32,
                    help="Head dimension for multi-head attention")
parser.add_argument("--lstm_dim", type=int, default=512,
                    help="LSTM hidden dimension")
parser.add_argument("--n_layer", type=int, default=3,
                    help="Number of layers in the network")
parser.add_argument("--condition_dim", type=int, default=256,
                    help="Condition dimension for conditional GAN")
parser.add_argument("--dependency_batchsize", type=int,
                    default=40, help="Batch size for dependency computations")
parser.add_argument("--args", type=int, nargs="+",
                    default=[-1, 48, 80, 4, 32], help="Arguments related to dimensions and heads")
parser.add_argument("--z_dim", type=int, default=128,
                    help="Dimension of latent vector z")
parser.add_argument("--g_dims", type=int, nargs="+",
                    help="Generator dimensions")
parser.add_argument("--d_dims", type=int, nargs="+",
                    default=[256, 256, 256, 128, 128, 128], help="Discriminator dimensions")
parser.add_argument("--gan_batchsize", type=int,
                    default=2500, help="Batch size for GAN training")
parser.add_argument("--gp_weight", type=int, default=10,
                    help="Gradient penalty weight for WGAN-GP")

args = parser.parse_args()
# Dynamically define g_dims
if args.g_dims is None:
    args.g_dims = [256, 256, 512, 512, 512, 512,
                   args.code_vocab_dim + args.label_vocab_dim + 2]
print(args)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))

# index_to_code = pickle.load(open("../../data/indexToCode.pkl", "rb"))
with open(f"data/{args.dataset}/{args.dataset_version}/meta.pkl", 'rb') as f:
    meta = pickle.load(f)
index_to_code = meta['itos']
# id_to_label = pickle.load(open("../../data/idToLabel.pkl", "rb"))
model = DependencyModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
generator = Generator(args).to(device)
generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=4e-6, weight_decay=1e-5)
checkpoint1 = torch.load(
    f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_synteg_dependency_model.pt", map_location=device)
checkpoint2 = torch.load(
    f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_synteg_condition_model.pt", map_location=device)
model.load_state_dict(checkpoint1['model_state_dict'])
optimizer.load_state_dict(checkpoint1['optimizer_state_dict'])
generator.load_state_dict(checkpoint2['generator'])
generator_optimizer.load_state_dict(checkpoint2['generator_optimizer'])

# # Add the labels to the index_to_code mapping
# for k, l in id_to_label.items():
#   index_to_code[args.code_vocab_dim+k] = f"Chronic Condition: {l}"


def sample_sequence(length, context, batch_size, device='cuda'):
    context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(
        0).repeat(batch_size, 1).to(device)
    ehr = context.unsqueeze(1).to(device)
    batch_ehr = torch.tensor(np.ones((batch_size, args.max_num_visit,
                             args.max_length_visit)) * args.vocab_dim, dtype=torch.long).to(device)
    batch_ehr[:, 0, 0] = args.code_vocab_dim + args.label_vocab_dim
    batch_lens = torch.zeros(
        (batch_size, args.max_num_visit, 1), dtype=torch.int).to(device)
    batch_lens[:, 0, 0] = 1
    with torch.no_grad():
        for j in range(length-1):
            for i in range(batch_size):
                codes = torch.nonzero(ehr[i, j]).squeeze(1)
                batch_ehr[i, j, 0:len(
                    codes)] = codes[0:args.max_length_visit]
                batch_lens[i, j] = min(len(codes), args.max_length_visit)
            condition_vector = model(batch_ehr, batch_lens, export=True)
            condition = condition_vector[:, j, :]
            z = torch.randn((batch_size, args.z_dim)).to(device)
            visit = generator(z, condition)
            visit = torch.bernoulli(visit).unsqueeze(1)
            ehr = torch.cat((ehr, visit), dim=1)
    ehr = ehr.cpu().detach().numpy()
    return ehr


def convert_ehr(ehrs, index_to_code=None):
    ehr_outputs = []
    for i in range(len(ehrs)):
        ehr = ehrs[i]
        ehr_output = []
        labels_output = ehr[1][args.code_vocab_dim:
                               args.code_vocab_dim+args.label_vocab_dim]
        if index_to_code is not None:
            labels_output = [index_to_code[idx + args.code_vocab_dim]
                             for idx in np.nonzero(labels_output)[0]]
        for j in range(2, len(ehr)):
            visit = ehr[j]
            visit_output = []
            indices = np.nonzero(visit)[0]
            end = False
            for idx in indices:
                if idx < args.code_vocab_dim:
                    visit_output.append(
                        index_to_code[idx] if index_to_code is not None else idx)
                elif idx == args.code_vocab_dim+args.label_vocab_dim+1:
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

# Generate a few sampled EHR for examinations
# stoken = np.zeros(args.vocab_dim)
# synthetic_ehrs = sample_sequence(args.max_num_visit, stoken, batch_size=3, device=device)
# synthetic_ehrs = convert_ehr(synthetic_ehrs, index_to_code)
# print("Sampled Synthetic EHRs: ")
# for i in range(3):
#    print("Labels: ")
#    print(synthetic_ehrs[i]['labels'])
#    print("Visits: ")
#    for v in synthetic_ehrs[i]['visits']:
#        print(v)
#    print("\n\n")


# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
count = 0
stoken = np.zeros(args.vocab_dim)
for i in tqdm(range(0, len(train_ehr_dataset), args.dependency_batchsize)):
    bs = min([len(train_ehr_dataset)-i, args.dependency_batchsize])
    batch_synthetic_ehrs = sample_sequence(
        args.max_num_visit, stoken, batch_size=bs, device=device)
    batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs)
    synthetic_ehr_dataset += batch_synthetic_ehrs
    # if len(synthetic_ehr_dataset) > 10000:
    #     pickle.dump(synthetic_ehr_dataset, open(
    #         f'{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_syntegDataset_{count}.pkl', 'wb'))
    #     synthetic_ehr_dataset = []
    #     count += 1

pickle.dump(synthetic_ehr_dataset, open(
    f'{args.LOG_DIR}/{args.dataset}_{args.dataset_version}__synteg.pkl', 'wb'))
