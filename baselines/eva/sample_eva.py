import torch
import pickle
import random
import numpy as np
import argparse
from tqdm import tqdm
from sklearn import metrics
from eva import Eva
from config import EVAConfig


config = EVAConfig()

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
parser.add_argument('--LOG_DIR', type=str, required=True)


args = parser.parse_args()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
  
train_data_path = f"data/{args.dataset}/{args.dataset_version}/train.pkl"
train_ehr_dataset = pickle.load(open(train_data_path, 'rb'))
with open(f"data/{args.dataset}/{args.dataset_version}/meta.pkl", 'rb') as f:
    meta = pickle.load(f)
index_to_code = meta['itos']

# train_ehr_dataset = pickle.load(open('../../data/trainDataset.pkl', 'rb'))
# test_ehr_dataset = pickle.load(open('../../data/testDataset.pkl', 'rb'))
# index_to_code = pickle.load(open("../../data/indexToCode.pkl", "rb"))
# id_to_label = pickle.load(open("../../data/idToLabel.pkl", "rb"))
# train_c = set([c for p in train_ehr_dataset for v in p['visits'] for c in v])
# test_ehr_dataset = [{'labels': p['labels'], 'visits': [[c for c in v if c in train_c] for v in p['visits']]} for p in test_ehr_dataset]

# # Add the labels to the index_to_code mapping
# for k, l in id_to_label.items():
#   index_to_code[config.code_vocab_size+k] = f"Chronic Condition: {l}"
  
# Add the labels to the index_to_code mapping
# index_to_code[config.code_vocab_size] = "Chronic Condition: Alzheimer or related disorders or senile"
# index_to_code[config.code_vocab_size+1] = "Chronic Condition: Heart Failure"
# index_to_code[config.code_vocab_size+2] = "Chronic Condition: Chronic Kidney Disease"
# index_to_code[config.code_vocab_size+3] = "Chronic Condition: Cancer"
# index_to_code[config.code_vocab_size+4] = "Chronic Condition: Chronic Obstructive Pulmonary Disease"
# index_to_code[config.code_vocab_size+5] = "Chronic Condition: Depression"
# index_to_code[config.code_vocab_size+6] = "Chronic Condition: Diabetes"
# index_to_code[config.code_vocab_size+7] = "Chronic Condition: Ischemic Heart Disease"
# index_to_code[config.code_vocab_size+8] = "Chronic Condition: Osteoporosis"
# index_to_code[config.code_vocab_size+9] = "Chronic Condition: rheumatoid arthritis and osteoarthritis (RA/OA)"
# index_to_code[config.code_vocab_size+10] = "Chronic Condition: Stroke/transient Ischemic Attack"


model = Eva(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

checkpoint = torch.load(f'{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_eva.pt', map_location=device)
print(f'load model from eva.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def convert_ehr(ehrs, index_to_code=None):
  ehr_outputs = []
  for i in range(len(ehrs)):
    ehr = ehrs[i]
    ehr_output = []
    labels_output = ehr[0][config.code_vocab_size:config.code_vocab_size+config.label_vocab_size]
    if index_to_code is not None:
      labels_output = [index_to_code[idx + config.code_vocab_size] for idx in np.nonzero(labels_output)[0]]
    for j in range(1, len(ehr)):
      visit = ehr[j]
      visit_output = []
      indices = np.nonzero(visit)
      if len(indices) > 0:
        indices = indices[0]
      else:
        continue
      end = False
      for idx in indices:
        if idx < config.code_vocab_size: 
          visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
        elif idx == config.code_vocab_size+config.label_vocab_size+1:
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

# Generate Synthetic EHR dataset
synthetic_ehr_dataset = []
for i in tqdm(range(0, len(train_ehr_dataset), config.batch_size)):
  bs = min([len(train_ehr_dataset)-i, config.batch_size])
  batch_synthetic_ehrs = model.sample(bs, device)
  batch_synthetic_ehrs = torch.bernoulli(batch_synthetic_ehrs)
  batch_synthetic_ehrs = convert_ehr(batch_synthetic_ehrs.detach().cpu().numpy())
  synthetic_ehr_dataset += batch_synthetic_ehrs

# pickle.dump(synthetic_ehr_dataset, open(f'../../results/datasets/evaDataset.pkl', 'wb'))
pickle.dump(synthetic_ehr_dataset, open(f"{args.LOG_DIR}/{args.dataset}_{args.dataset_version}_eva.pkl", 'wb'))
