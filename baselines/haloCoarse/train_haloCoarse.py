import os
import torch
import numpy as np
import random
import argparse
import pickle
import time
import json
from tqdm import tqdm
from haloCoarse import HALOCoarseModel

# Argument parsing
parser = argparse.ArgumentParser(description='Train the HALOCoarse model')
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
# Positional and context configurations
parser.add_argument("--n_positions", type=int, default=56,
                    help="Number of positional embeddings")
parser.add_argument("--n_ctx", type=int, default=48,
                    help="Context window size")
# Embedding and model parameters
parser.add_argument("--n_embd", type=int, default=768,
                    help="Embedding size")
parser.add_argument("--n_layer", type=int, default=12,
                    help="Number of layers")
parser.add_argument("--n_head", type=int, default=12,
                    help="Number of attention heads")
parser.add_argument("--layer_norm_epsilon", type=float,
                    default=1e-5, help="Layer normalization epsilon")
parser.add_argument("--initializer_range", type=float,
                    default=0.02, help="Initializer range for model parameters")
# Training parameters
parser.add_argument("--batch_size", type=int,
                    default=128, help="Batch size for training")
parser.add_argument("--epoch", type=int, default=25,
                    help="Number of training epochs")
parser.add_argument("--pos_loss_weight", type=float,
                    default=None, help="Positive loss weight")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()
# Print parsed arguments
print(args)

# Set a fixed seed for reproducibility
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

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
val_data_path = f"data/{args.dataset}/{args.dataset_version}/val.pkl"
val_ehr_dataset = pickle.load(open(val_data_path, 'rb'))

# train_ehr_dataset = pickle.load(open('../../data/trainDataset.pkl', 'rb'))
# val_ehr_dataset = pickle.load(open('../../data/valDataset.pkl', 'rb'))

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
  for i, p in enumerate(ehr):
    visits = p['visits']
    for j, v in enumerate(visits):
      batch_ehr[i,j+2][v] = 1
      batch_mask[i,j+2] = 1
    batch_ehr[i,1,args.code_vocab_size:args.code_vocab_size+args.label_vocab_size] = np.array(p['labels']) # Set the patient labels
    batch_ehr[i,len(visits)+1,args.code_vocab_size+args.label_vocab_size+1] = 1 # Set the final visit to have the end token
    batch_ehr[i,len(visits)+2:,args.code_vocab_size+args.label_vocab_size+2] = 1 # Set the rest to the padded visit token
  
  batch_mask[:,1] = 1 # Set the mask to cover the labels
  batch_ehr[:,0,args.code_vocab_size+args.label_vocab_size] = 1 # Set the first visits to be the start token
  batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
  return batch_ehr, batch_mask

def shuffle_training_data(train_ehr_dataset):
  np.random.shuffle(train_ehr_dataset)

model = HALOCoarseModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# if os.path.exists("../../save/haloCoarse_model"):
#   print("Loading previous model")
#   checkpoint = torch.load('../../save/haloCoarse_model', map_location=torch.device(device))
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
    
    batch_ehr, batch_mask = get_batch(i, args.batch_size, 'train')
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
    
    optimizer.zero_grad()
    loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=args.pos_loss_weight)
    loss.backward()
    optimizer.step()
    free, total = torch.cuda.mem_get_info(device)
    gpu_memory = (total - free)/ (1024 ** 2)  # In MB
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    
    if i % (100*args.batch_size) == 0:
      print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss))
    if i % (100*args.batch_size) == 0:
      if i == 0:
        continue
    
      model.eval()
      with torch.no_grad():
        val_l = []
        for v_i in range(0, len(val_ehr_dataset), args.batch_size):
          batch_ehr, batch_mask = get_batch(v_i, args.batch_size, 'valid')
          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)
  
          val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=args.pos_loss_weight)
          val_l.append((val_loss).cpu().detach().numpy())
          
        cur_val_loss = np.mean(val_l)
        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
        if cur_val_loss < global_loss:
          train_end_time = time.time()
          global_loss = cur_val_loss
          state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': i
            }
          torch.save(state, os.path.join(args.LOG_DIR, f'{args.dataset}_{args.dataset_version}_haloCoarse.pt'))
          print('\n------------ Save best model ------------\n')
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
