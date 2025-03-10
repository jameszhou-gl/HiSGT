import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from synteg import DependencyModel


# Set a fixed seed for reproducibility
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
# Print parsed arguments
print(args)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# train_ehr_dataset = pickle.load(open('../../data/trainDataset.pkl', 'rb'))
# val_ehr_dataset = pickle.load(open('../../data/valDataset.pkl', 'rb'))
train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))
val_data_path = f"data/{args.dataset}/{args.dataset_version}/val.pkl"
val_ehr_dataset = pickle.load(open(val_data_path, 'rb'))


def get_batch(loc, batch_size, mode):
    # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
    #   Where each patient P is [V_1, V_2, ... , V_j]
    #     Where each visit V is [C_1, C_2, ... , C_k]
    #   And where each Label L is a binary vector [L_1 ... L_11]
    if mode == 'train':
        ehr = train_ehr_dataset[loc:loc+batch_size]
    elif mode == 'valid':
        ehr = val_ehr_dataset[loc:loc+batch_size]
    else:
        ehr = test_ehr_dataset[loc:loc+batch_size]

    batch_ehr = np.zeros(
        (len(ehr), args.max_num_visit, args.max_length_visit))
    # Initialize each code to the padding code
    batch_ehr[:, :, :] = args.vocab_dim-1
    batch_lens = np.ones((len(ehr), args.max_num_visit, 1))
    batch_mask = np.zeros((len(ehr), args.max_num_visit, 1))
    batch_num_visits = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits']
        for j, v in enumerate(visits):
            batch_mask[i, j+2] = 1
            batch_lens[i, j+2] = len(v) + 1
            for k, c in enumerate(v):
                assert c < args.code_vocab_dim, f"Invalid medical code {c} exceeds vocab_dim {args.code_vocab_dim}"
                batch_ehr[i, j+2, k+1] = c
        # Set the last code in the last visit to be the end record code
        batch_ehr[i, j+2, len(v)+1] = args.code_vocab_dim + \
            args.label_vocab_dim + 1
        batch_lens[i, j+2] = len(v) + 2
        for l_idx, l in enumerate(np.nonzero(p['labels'])[0]):
            batch_ehr[i, 1, l_idx+1] = args.code_vocab_dim + l
            batch_lens[i, 1] = l_idx+2
        batch_num_visits[i] = len(visits)

    batch_mask[:, 1] = 1  # Set the mask to cover the labels
    # Set the first code in each visit to be the start/class token
    batch_ehr[:, :, 0] = args.code_vocab_dim + args.label_vocab_dim
    # Shift the mask to match the shifted labels and predictions the model will return
    batch_mask = batch_mask[:, 1:, :]

    return batch_ehr, batch_lens, batch_mask, batch_num_visits


def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)


EPOCHS = 50
LR = 1e-4
model = DependencyModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# if os.path.exists("../../save/synteg_dependency_model"):
#     print("Loading previous model")
#     checkpoint = torch.load(
#         "../../save/synteg_dependency_model", map_location=torch.device(device))
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])

# Train
global_loss = 1e10
for e in tqdm(range(EPOCHS)):
    shuffle_training_data(train_ehr_dataset)
    for i in range(0, len(train_ehr_dataset), args.dependency_batchsize):
        model.train()

        batch_ehr, batch_lens, batch_mask, _ = get_batch(
            i, args.dependency_batchsize, 'train')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.int).to(device)
        batch_lens = torch.tensor(batch_lens, dtype=torch.int).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.int).to(device)
        assert torch.max(batch_ehr) < args.vocab_dim + \
            1, "Index out of range in batch_ehr"
        assert batch_lens.max() <= args.max_length_visit, "Sequence length exceeds max_length_visit"
        optimizer.zero_grad()
        diagnosis_output = model(batch_ehr, batch_lens)
        labels = torch.sum(nn.functional.one_hot(
            batch_ehr.long(), num_classes=args.vocab_dim+1), dim=2).float()
        labels = labels[..., 1:, :-1].contiguous()
        
        labels = torch.clamp(labels, min=0, max=1)
        batch_mask = torch.clamp(batch_mask, min=0, max=1)
        
        diagnosis_output = diagnosis_output * batch_mask
        labels = labels * batch_mask
        bce = nn.BCELoss()
        loss = bce(diagnosis_output, labels)
        loss.backward()
        optimizer.step()

        if i % (500*args.dependency_batchsize) == 0:
            print("Epoch %d, Iter %d: Training Loss:%.6f" % (e, i, loss))
        if i % (500*args.dependency_batchsize) == 0:
            model.eval()
            with torch.no_grad():
                val_l = []
                for v_i in range(0, len(val_ehr_dataset), args.dependency_batchsize):
                    batch_ehr, batch_lens, batch_mask, _ = get_batch(
                        v_i, args.dependency_batchsize, 'valid')
                    batch_ehr = torch.tensor(
                        batch_ehr, dtype=torch.int).to(device)
                    batch_lens = torch.tensor(
                        batch_lens, dtype=torch.int).to(device)
                    batch_mask = torch.tensor(
                        batch_mask, dtype=torch.int).to(device)
                    batch_mask = torch.clamp(batch_mask, min=0, max=1)

                    diagnosis_output = model(batch_ehr, batch_lens)
                    labels = torch.sum(nn.functional.one_hot(
                        batch_ehr.long(), num_classes=args.vocab_dim+1), dim=2).float()
                    labels = labels[..., 1:, :-1].contiguous()
                    diagnosis_output = diagnosis_output * batch_mask
                    labels = labels * batch_mask
                    labels = torch.clamp(labels, min=0, max=1)
                    bce = nn.BCELoss()
                    val_loss = bce(diagnosis_output, labels)
                    val_l.append(val_loss.item())

            cur_val_loss = np.mean(val_l)
            print("Epoch %d Validation Loss:%.7f" % (e, cur_val_loss))
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': e,
                    'loss': cur_val_loss
                }
                torch.save(state, os.path.join(args.LOG_DIR, f'{args.dataset}_{args.dataset_version}_synteg_dependency_model.pt'))
                print("Best model saved.")
