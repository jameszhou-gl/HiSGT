'''
    code is modified based on HALO (https://github.com/btheodorou99/HALO_Inpatient)
'''
import pdb
import os
import csv
import yaml
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Set a fixed random seed for reproducibility
SEED = 1337
np.random.seed(SEED)

# Set paths to MIMIC-III data files
mimic_dir = os.path.dirname(__file__)
admission_file = f"{mimic_dir}/ADMISSIONS.csv"
diagnosis_file = f"{mimic_dir}/DIAGNOSES_ICD.csv"


# Load and preprocess admission and diagnosis data
print("Loading CSVs into DataFrames")
admission_df = pd.read_csv(admission_file, dtype=str)
admission_df['ADMITTIME'] = pd.to_datetime(admission_df['ADMITTIME'])
admission_df = admission_df.sort_values('ADMITTIME').reset_index(drop=True)

diagnosis_df = pd.read_csv(diagnosis_file, dtype=str).set_index("HADM_ID")
diagnosis_df = diagnosis_df[diagnosis_df['ICD9_CODE'].notnull()][['ICD9_CODE']]

# Construct dataset: group admissions and diagnoses by patient
print("Building dataset")
data = {}
hadms_with_empty_diagnoses = []
for row in tqdm(admission_df.itertuples(), total=admission_df.shape[0]):
    hadm_id, subject_id = row.HADM_ID, row.SUBJECT_ID
    # ! modify here. we prefer preserving raw data, instead of removing duplicates with set() in original HALO code (https://github.com/btheodorou99/HALO_Inpatient/blob/75b96f5692c1cd55fa88eb3deb31754fc2bc0807/build_dataset.py#L30)
    diagnoses = list(
        diagnosis_df.loc[[hadm_id], "ICD9_CODE"]) if hadm_id in diagnosis_df.index else []
    if diagnoses == []:
        hadms_with_empty_diagnoses.append(hadm_id)
    data.setdefault(subject_id, {'visits': []})['visits'].append(diagnoses)
print(f'hadms_with_empty_diagnoses: {hadms_with_empty_diagnoses}')
# Build vocabulary mapping for ICD9 codes
# add sorted to make sure remove the randomness from set()
all_codes = sorted(set(c for patient in data.values()for visit in patient['visits'] for c in visit))
np.random.shuffle(all_codes)
# code_to_index: Dictionary mapping ICD9_CODE to unique integer index
code_to_index = {code: idx for idx, code in enumerate(all_codes)}
cur_len = len(code_to_index)
# special_toekns = {'START_RECORD': cur_len, 'START_VISIT': cur_len +
#                   1, 'END_VISIT': cur_len+2, 'END_RECORD': cur_len+3}
# code_to_index.update(special_toekns)
index_to_code = {idx: code for code, idx in code_to_index.items()}
vocab_size = len(code_to_index)
print(f"VOCAB SIZE: {vocab_size}")
# print(f"Special tokens: {len(special_toekns)}")

# ! Load and process label definitions, copy from HALO
print("Adding labels")
with open(f"{mimic_dir}/hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
    definitions = yaml.safe_load(definitions_file)
# Map ICD9 codes to group labels
code_to_group = {
    code: group for group, details in definitions.items()
    if details['use_in_benchmark']
    for code in details['codes']
}
# `id_to_group` structure as a dictionary instead of a list in the original HALO code
id_to_group = {idx: group for idx, group in enumerate(
    sorted([k for k in definitions if definitions[k]['use_in_benchmark']]))}
# group_to_id: mapping group to unique ID
group_to_id = {group: idx for idx, group in id_to_group.items()}
# Assign labels based on ICD9 codes for each patient
for patient in data.values():
    label_vector = np.zeros(len(group_to_id))
    for visit in patient['visits']:
        for code in visit:
            if code in code_to_group:
                label_vector[group_to_id[code_to_group[code]]] = 1
    patient['labels'] = label_vector
# Convert ICD9 codes in each visit to vocabulary indices
print("Converting visits to indices")
for patient in data.values():
    # ! one difference from raw HALO code: we use `for code in visit` instead of `for code in set(visit)`
    # https://github.com/btheodorou99/HALO_Inpatient/blob/75b96f5692c1cd55fa88eb3deb31754fc2bc0807/build_dataset.py#L86
    # set(visit) would remove duplicates in each visit and also terribly change the order of ICD9 codes.
    patient['visits'] = [[code_to_index[code]
                          for code in visit] for visit in patient['visits']]

# Display dataset statistics
# ! `data` structure:
# {
#   subject_id: {
#       'visits': [[ICD9_CODE indices for visit 1], [ICD9_CODE indices for visit 2], ...],
#       'labels': binary vector indicating the presence of each diagnostic group
#   }
# }
num_visits = [len(patient['visits']) for patient in data.values()]
visit_lengths = [len(visit) for patient in data.values()
                 for visit in patient['visits']]
print(f"MAX VISITS PER RECORD: {max(num_visits)}, AVG VISITS PER RECORD: {np.mean(num_visits)}, MIN VISITS PER RECORD: {min(num_visits)}")
print(f"MAX ICD9 CODES PER VISIT: {max(visit_lengths)}, AVG CODES PER VISIT: {np.mean(visit_lengths)}, MIN ICD9 CODES PER VISIT: {min(visit_lengths)}")
print(f"EMPTY VISITS: {sum(1 for visit_length in visit_lengths if visit_length == 0)}")
print(f"TOTAL RECORDS: {len(data)}, LONGITUDINAL RECORDS: {sum(1 for subj_info in data.values() if len(subj_info['visits']) > 1)}")

# Split into train, validation, and test sets
print("Splitting datasets")
data_list = list(data.values())
train_data, test_data = train_test_split(
    data_list, test_size=0.2, random_state=SEED, shuffle=True)
train_data, val_data = train_test_split(
    train_data, test_size=0.1, random_state=SEED, shuffle=True)

# Save processed data and mappings
print("Saving datasets and mappings")
pickle.dump(train_data, open(f"{mimic_dir}/train.pkl", "wb"))
pickle.dump(val_data, open(f"{mimic_dir}/val.pkl", "wb"))
pickle.dump(test_data, open(f"{mimic_dir}/test.pkl", "wb"))
with open(f"{mimic_dir}/idToLabel.json", "w") as file:
    json.dump(id_to_group, file)

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': index_to_code,
    'stoi': code_to_index,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

with open(os.path.join(os.path.dirname(__file__), "vocab_stoi.csv"), mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header (optional)
    writer.writerow(["String", "Token Index"])
    for token, index in code_to_index.items():
        writer.writerow([token, index])
