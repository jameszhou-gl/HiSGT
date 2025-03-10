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
admission_file = f"{mimic_dir}/admissions.csv"
diagnosis_file = f"{mimic_dir}/diagnoses_icd.csv"

# Load and preprocess admission and diagnosis data
print("Loading CSVs into DataFrames")
admission_df = pd.read_csv(admission_file, dtype=str)
admission_df['admittime'] = pd.to_datetime(admission_df['admittime'])
admission_df = admission_df.sort_values('admittime').reset_index(drop=True)

diagnosis_df = pd.read_csv(diagnosis_file, dtype=str).set_index("hadm_id")
diagnosis_df = diagnosis_df[(diagnosis_df['icd_code'].notnull()) & (diagnosis_df['icd_version'] == '9')][['icd_code']]
print(f'Number of admissions: {len(admission_df)}')
print(f'Number of diagnoses (ICD-9 only): {len(diagnosis_df)}')

# Construct dataset: group admissions and diagnoses by patient
print("Building dataset")
# Step 1: Group and aggregate diagnoses by `hadm_id`
diagnoses_aggregated = diagnosis_df.groupby(
    'hadm_id')['icd_code'].apply(list).reset_index()
# print(f"Aggregated diagnoses: {diagnoses_aggregated.head()}")

# Step 2: Merge aggregated diagnoses with admission data
admission_df = admission_df.merge(
    diagnoses_aggregated, how='left', on='hadm_id')
admission_df['icd_code'] = admission_df['icd_code'].apply(
    lambda x: x if isinstance(x, list) else [])
# Filter out admissions with no diagnoses
admission_df = admission_df[admission_df['icd_code'].apply(len) > 0]
print(f"Filtered admissions with no diagnoses. Remaining admissions: {len(admission_df)}")

# Step 3: Group admissions and diagnoses by `subject_id`
grouped_data = admission_df.groupby('subject_id').apply(
    lambda x: {'visits': x['icd_code'].tolist()}
).to_dict()
# Remove patients with empty codes
print("Removing patients with empty codes...")
grouped_data = {subject_id: patient_data for subject_id, patient_data in grouped_data.items()
                if any(len(visit) > 0 for visit in patient_data['visits'])}

# Step 4: Handle patients with no diagnoses
hadms_with_empty_diagnoses = admission_df[admission_df['icd_code'].apply(
    len) == 0]['hadm_id'].tolist()

# Print stats
# print(f'hadms_with_empty_diagnoses: {hadms_with_empty_diagnoses}')

# Step 5: Build vocabulary mapping
all_codes = sorted(set(c for patient in grouped_data.values()
                   for visit in patient['visits'] for c in visit))
np.random.shuffle(all_codes)
code_to_index = {code: idx for idx, code in enumerate(all_codes)}
index_to_code = {idx: code for code, idx in code_to_index.items()}
vocab_size = len(code_to_index)

print(f"VOCAB SIZE: {vocab_size}")

# Add labels for each patient
print("Adding labels")
with open(f"{mimic_dir}/hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
    definitions = yaml.safe_load(definitions_file)

# Map ICD codes to group labels
code_to_group = {
    code: group for group, details in definitions.items()
    if details['use_in_benchmark']
    for code in details['codes']
}
id_to_group = {idx: group for idx, group in enumerate(
    sorted([k for k in definitions if definitions[k]['use_in_benchmark']]))}
group_to_id = {group: idx for idx, group in id_to_group.items()}

# Assign labels based on ICD codes for each patient
for patient in grouped_data.values():
    label_vector = np.zeros(len(group_to_id))
    for visit in patient['visits']:
        for code in visit:
            if code in code_to_group:
                label_vector[group_to_id[code_to_group[code]]] = 1
    patient['labels'] = label_vector

# Convert visits to indices
print("Converting visits to indices")
for patient in grouped_data.values():
    patient['visits'] = [[code_to_index[code] for code in visit]
                         for visit in patient['visits']]

# Calculate the initial number of patients
initial_patient_count = len(grouped_data)

# Filter out patients whose total ICD codes exceed 768
print("Excluding patients with total ICD codes exceeding 768...")
grouped_data = {subject_id: patient_data for subject_id, patient_data in grouped_data.items()
                if sum(len(visit) for visit in patient_data['visits']) <= 768}

# Calculate the final number of patients
final_patient_count = len(grouped_data)

# Calculate and print the number of removed patients
removed_patient_count = initial_patient_count - final_patient_count
print(f"Number of patients removed: {removed_patient_count}")
print(f"Remaining patients: {final_patient_count}")

num_visits = [len(patient['visits']) for patient in grouped_data.values()]
visit_lengths = [len(visit) for patient in grouped_data.values()
                 for visit in patient['visits']]

print(f"MAX VISITS PER RECORD: {max(num_visits)}, AVG VISITS PER RECORD: {np.mean(num_visits)}, MIN VISITS PER RECORD: {min(num_visits)}")
print(f"MAX ICD9 CODES PER VISIT: {max(visit_lengths)}, AVG CODES PER VISIT: {np.mean(visit_lengths)}, MIN ICD9 CODES PER VISIT: {min(visit_lengths)}")
print(f"EMPTY VISITS: {sum(1 for visit_length in visit_lengths if visit_length == 0)}")
print(f"TOTAL RECORDS: {len(grouped_data)}, LONGITUDINAL RECORDS: {sum(1 for subj_info in grouped_data.values() if len(subj_info['visits']) > 1)}")
# Calculate the maximum number of ICD codes per patient
print("Calculating maximum number of ICD codes per patient...")
max_icd_codes_per_patient = max(
    sum(len(visit) for visit in patient['visits']) for patient in grouped_data.values()
)
avg_icd_codes_per_patient = np.mean([
    sum(len(visit) for visit in patient['visits']) for patient in grouped_data.values()
])
min_icd_codes_per_patient = min(
    sum(len(visit) for visit in patient['visits']) for patient in grouped_data.values()
)

print(f"MAX ICD CODES PER RECORD: {max_icd_codes_per_patient}")
print(f"AVG ICD CODES PER RECORD: {avg_icd_codes_per_patient}")
print(f"MIN ICD CODES PER RECORD: {min_icd_codes_per_patient}")

# Save the data and vocabulary
print("Splitting datasets")
train_data, test_data = train_test_split(
    list(grouped_data.values()), test_size=0.2, random_state=SEED, shuffle=True)
train_data, val_data = train_test_split(
    train_data, test_size=0.1, random_state=SEED, shuffle=True)

# Save to disk
pickle.dump(train_data, open(f"{mimic_dir}/train.pkl", "wb"))
pickle.dump(val_data, open(f"{mimic_dir}/val.pkl", "wb"))
pickle.dump(test_data, open(f"{mimic_dir}/test.pkl", "wb"))

with open(f"{mimic_dir}/idToLabel.json", "w") as file:
    json.dump(id_to_group, file)

meta = {'vocab_size': vocab_size, 'itos': index_to_code, 'stoi': code_to_index}
with open(f'{mimic_dir}/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
    
with open(os.path.join(mimic_dir, "vocab_stoi.csv"), mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header (optional)
    writer.writerow(["String", "Token Index"])
    for token, index in code_to_index.items():
        writer.writerow([token, index])
