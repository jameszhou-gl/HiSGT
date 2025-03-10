import os
import torch
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from eva import Eva
import pdb
# pdb.set_trace()


# Set a fixed seed for reproducibility
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
# args = HALOConfig()

# Argument parsing
parser = argparse.ArgumentParser(description='Train the EVA model')
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
parser.add_argument("--n_ctx", type=int, default=57,
                    help="Context window size")
parser.add_argument("--n_embd", type=int, default=768,
                    help="Embedding size")
parser.add_argument("--latent_dim", type=int,
                    default=32, help="Latent dimension size")
parser.add_argument("--n_lstm_layer", type=int,
                    default=1, help="Number of LSTM layers")
parser.add_argument("--n_conv1d_layer", type=int, default=3,
                    help="Number of 1D convolutional layers")
parser.add_argument("--n_deconv_layer", type=int,
                    default=4, help="Number of deconvolution layers")
parser.add_argument("--dilation_factor", type=int, default=2,
                    help="Dilation factor for convolution layers")
parser.add_argument("--deconv_factor", type=int,
                    default=3, help="Deconvolution factor")
parser.add_argument("--batch_size", type=int,
                    default=128, help="Batch size for training")
parser.add_argument("--prob_batch_size", type=int, default=4,
                    help="Batch size for probability computation")
parser.add_argument("--epoch", type=int, default=50,
                    help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--pos_loss_weight", type=float,
                    default=None, help="Positive loss weight")


args = parser.parse_args()
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

max_visits = max(len(p['visits']) for p in train_ehr_dataset)
required_ctx = max_visits + 3  # Visits + special tokens
if args.n_ctx < required_ctx:
    raise ValueError(f"n_ctx ({args.n_ctx}) is too small. It must be at least {required_ctx}.")
  
def get_batch(loc, batch_size, mode):
  # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
  #   Where each patient P is [V_1, V_2, ... , V_j]
  #     Where each visit V is [C_1, C_2, ... , C_k]
  #   And where each Label L is a binary vector [L_1 ... L_n]
  if mode == 'train':
    ehr = train_ehr_dataset[loc:loc+batch_size]
  elif mode == 'valid':
    ehr = val_ehr_dataset[loc:loc+batch_size]
  else:
    ehr = test_ehr_dataset[loc:loc+batch_size]
    
  batch_ehr = np.zeros((len(ehr), args.n_ctx, args.total_vocab_size))
  batch_mask = np.zeros((len(ehr), args.n_ctx, 1))
  batch_lens = np.zeros(len(ehr))
  for i, p in enumerate(ehr):
    visits = p['visits']
    batch_lens[i] = len(visits)
    for j, v in enumerate(visits):
      batch_ehr[i,j+2][v] = 1
      batch_mask[i,j+2] = 1
    batch_ehr[i,1,args.code_vocab_size:args.code_vocab_size+args.label_vocab_size] = np.array(p['labels']) # Set the patient labels
    batch_ehr[i,len(visits)+1,args.code_vocab_size+args.label_vocab_size+1] = 1 # Set the final visit to have the end token
    batch_ehr[i,len(visits)+2:,args.code_vocab_size+args.label_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_mask[:,1] = 1 # Set the mask to cover the labels
  batch_ehr[:,0,args.code_vocab_size+args.label_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_lens, batch_mask

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

model = Eva(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if os.path.exists("../../save/eva_model"):
  print("Loading previous model")
  checkpoint = torch.load('../../save/eva_model', map_location=torch.device(device))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])

# Train
global_loss = 1e10
kl_schedule = [0.1, 0.15, 0.25, 0.325, 0.5, 0.75, 0.9, 1.0, 1.0, 1.0]
for e in tqdm(range(args.epoch)):
  klw = kl_schedule[e] if e < len(kl_schedule) else 1
  shuffle_training_data(train_ehr_dataset)
  for i in range(0, len(train_ehr_dataset), args.batch_size):
    model.train()
    
    batch_ehr, batch_lens, batch_mask = get_batch(i, args.batch_size, 'train')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

    optimizer.zero_grad()
    loss, _, _ = model(batch_ehr, batch_lens, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=args.pos_loss_weight, kl_weight=klw)
    loss.backward()
    optimizer.step()
    
    if i % (100*args.batch_size) == 0:
      print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
    if i % (200*args.batch_size) == 0:
      if i == 0:
        continue
    
      model.eval()
      with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_ehr_dataset), args.batch_size):
          batch_ehr, batch_lens, batch_mask = get_batch(v_i, args.batch_size, 'valid')
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

          val_loss, _, _ = model(batch_ehr, batch_lens, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=args.pos_loss_weight, kl_weight=klw)
          val_l.append((val_loss).cpu().detach().numpy())
          
        cur_val_loss = np.mean(val_l)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
        if cur_val_loss < global_loss:
          global_loss = cur_val_loss
          state = {
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'epoch': e,
              'loss': cur_val_loss
          }
          torch.save(state, os.path.join(args.LOG_DIR, f'{args.dataset}_{args.dataset_version}_eva.pt'))
          print("Best model saved.")