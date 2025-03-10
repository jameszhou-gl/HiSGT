import os
import torch
import numpy as np
import random
import pickle
import time
import json
import argparse
from tqdm import tqdm
from HALO import HALOModel
# from baselines.halo.args import HALOConfig
# import pdb
# pdb.set_trace()

# Set a fixed seed for reproducibility
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
# args = HALOConfig()

# Argument parsing
parser = argparse.ArgumentParser(description='Train the HALO model')
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
parser.add_argument('--LOG_DIR', type=str, required=True)


args = parser.parse_args()
# Print parsed arguments
print(args)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))
val_data_path = f"data/{args.dataset}/{args.dataset_version}/val.pkl"
val_ehr_dataset = pickle.load(open(val_data_path, 'rb'))
# total_visit = []
# for i, train_ehr in enumerate(train_ehr_dataset):
#     visits = train_ehr['visits']
#     for visit in visits:
#         total_visit.extend(visit)
# total_unique_visit = list(set(total_visit))
# print(len(total_unique_visit))
# print(max(total_unique_visit))

# total_visit = total_unique_visit
# for i, val_ehr in enumerate(val_ehr_dataset):
#     visits = val_ehr['visits']
#     for visit in visits:
#         total_visit.extend(visit)
# total_unique_visit = list(set(total_visit))
# print(len(total_unique_visit))
# print(max(total_unique_visit))


def get_batch(loc, batch_size, mode):
    # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
    #   Where each patient P is [V_1, V_2, ... , V_j]
    #     Where each visit V is [C_1, C_2, ... , C_k]
    #   And where each Label L is a binary vector [L_1 ... L_n]
    # example of train_ehr_dataset[0]
    # {'visits': [[3905, 803, 4425, 5361, 5237]],
    # 'labels': array([0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}

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
        try:
            for j, v in enumerate(visits):
                # '+2' is because the first two 'visits' are left for start and label codes
                batch_ehr[i, j+2][v] = 1
                batch_mask[i, j+2] = 1
            batch_ehr[i, 1, args.code_vocab_size:args.code_vocab_size +
                      # Set the patient labels
                      args.label_vocab_size] = np.array(p['labels'])
            # Set the final visit to have the end token
            batch_ehr[i, len(visits)+1, args.code_vocab_size +
                      args.label_vocab_size+1] = 1
            # Set the rest to the padded visit token
            batch_ehr[i, len(visits)+2:, args.code_vocab_size +
                      args.label_vocab_size+2] = 1
        except Exception as e:
            print(e)
            print(i)

    batch_mask[:, 1] = 1  # Set the mask to cover the labels
    # Set the first visits to be the start token
    batch_ehr[:, 0, args.code_vocab_size+args.label_vocab_size] = 1
    # Shift the mask to match the shifted labels and predictions the model will return
    batch_mask = batch_mask[:, 1:, :]
    # ? what's the purpose of batch_mask, may use in Supplementary 2.3
    # ! Note about batch_mask:
    # The following section uses ehr_masks to mask certain parts of the data
    # during loss calculation. The mask's first element corresponds to patient
    # demographics or labels, while the subsequent elements represent actual visits.
    #
    # If we set the first element of the mask to 1, we are including the label
    # in the prediction process, which may not be necessary because the labels
    # are typically fixed attributes like age or gender. These labels serve as
    # inputs for generating future visits but should not be predicted themselves.
    #
    # Hence, it would be better to set the first element of the mask to 0 so
    # that the loss is calculated only for the predictions of actual visits.
    # This ensures that the demographic data is used as input and not mistakenly
    # treated as part of the output to be predicted.
    # ! batch_ehr would be like Fig.2 in the paper
    return batch_ehr, batch_mask


def shuffle_training_data(train_ehr_dataset):
    np.random.shuffle(train_ehr_dataset)


model = HALOModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
# if os.path.exists("./save/halo_model"):
#     print("Loading previous model")
#     checkpoint = torch.load('./save/halo_model', map_location=torch.device(device))
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])

# Train
train_start_time = time.time()
global_loss = 1e10
for epoch in tqdm(range(args.train_epochs)):
    epoch_start_time = time.time()
    shuffle_training_data(train_ehr_dataset)
    for i in range(0, len(train_ehr_dataset), args.batch_size):
        model.train()

        batch_ehr, batch_mask = get_batch(i, args.batch_size, 'train')
        batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
        batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr,
                           ehr_masks=batch_mask, pos_loss_weight=args.pos_loss_weight)
        loss.backward()
        optimizer.step()
        free, total = torch.cuda.mem_get_info(device)
        gpu_memory = (total - free) / (1024 ** 2)  # In MB
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)


        if i % (500*args.batch_size) == 0:
            print("Epoch %d, Iter %d: Training Loss:%.6f" % (epoch, i, loss * 8))
        if i % (500*args.batch_size) == 0:
            if i == 0:
                continue

            model.eval()
            with torch.no_grad():
                val_l = []
                for v_i in range(0, len(val_ehr_dataset), args.batch_size):
                    batch_ehr, batch_mask = get_batch(
                        v_i, args.batch_size, 'valid')
                    batch_ehr = torch.tensor(
                        batch_ehr, dtype=torch.float32).to(device)
                    batch_mask = torch.tensor(
                        batch_mask, dtype=torch.float32).to(device)

                    val_loss, _, _ = model(batch_ehr, position_ids=None, ehr_labels=batch_ehr,
                                           ehr_masks=batch_mask, pos_loss_weight=args.pos_loss_weight)
                    val_l.append((val_loss).cpu().detach().numpy())

                cur_val_loss = np.mean(val_l)
                print("Epoch %d Validation Loss:%.7f" % (epoch, cur_val_loss))
                if cur_val_loss < global_loss:
                    train_end_time = time.time()
                    global_loss = cur_val_loss
                    state = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': cur_val_loss
                    }
                    torch.save(state, os.path.join(args.LOG_DIR, f'{args.dataset}_{args.dataset_version}_halo.pt'))
                    print("Best model saved.")
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch} Time: {epoch_time:.2f} seconds")

# Post-training metrics
total_training_time = train_end_time - train_start_time
# Save metrics to JSON
computational_metrics = {
    "training_time_seconds": total_training_time,
    "training_gpu_memory_mb": gpu_memory,
    "total_parameters": num_params,
    "trainable_parameters": trainable_params,
    "batch_size": args.batch_size,
    "epochs": args.train_epochs,
    "learning_rate": args.learning_rate
}

with open(os.path.join(args.LOG_DIR, f"{args.dataset}_{args.dataset_version}_computational_cost.json"), "w") as f:
    json.dump(computational_metrics, f, indent=4)

print(f"Training Time: {total_training_time:.2f} seconds")
print(f"Training GPU Memory Used: {gpu_memory:.2f} MB")
print(f"Total Parameters: {num_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Saved computational metrics to {args.LOG_DIR}")
