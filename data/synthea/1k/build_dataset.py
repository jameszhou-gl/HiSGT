import argparse
from collections import defaultdict
import os
import csv
import yaml
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from preprocessing import OMOP_to_ICD9_conversion

import warnings
warnings.filterwarnings('ignore')

# Set a fixed random seed for reproducibility
SEED = 1337
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Preprocessing')
parser.add_argument('--dataset', type=str, default='synthea', help="Dataset name")
parser.add_argument('--dataset_version', type=str, default='1k', help="Dataset version")

args = parser.parse_args()
print(args)

current_directory = os.path.dirname(os.path.abspath(__file__))

# Load configuration file
print("Loading config file")
with open("config.yml") as config_file:
    config = yaml.safe_load(config_file)
    print(config)

# Get csv.active value
db_source = config["app"]["data_source"]
print(f"db_source: {db_source}")
if db_source=="csv":
    print("Preprocessing CSVs into DataFrames")
    # visit_df = pd.read_csv("synthea1k_ICD9.csv").head(100)
    visit_df = OMOP_to_ICD9_conversion(args=args, db_name="csv", db_config=config, save_csv=True)
elif db_source=="postgres":
    ## connect to a db
    print("Preprocessing DB into DataFrames")
    visit_df = OMOP_to_ICD9_conversion(db_name="postgres", db_config=config)
elif db_source=="sql":
    ## connect to a db
    print("Preprocessing DB into DataFrames")
    visit_df = OMOP_to_ICD9_conversion(db_name="sql", db_config=config)


if(visit_df is None):
    print("Error: visit_df is None")
    exit(1)
else:
    print("Building dataset")
    data = defaultdict(lambda: {'visits': []})
    hadms_with_empty_diagnoses = []
    hadms_with_empty_drugs = []

    for row in tqdm(visit_df.itertuples(index=False), total=visit_df.shape[0]):
        hadm_id, subject_id, icd9_code, drug_code = row.visit_occurrence_id, row.person_id, getattr(row, 'ICD9_CODE', None), getattr(row, 'drug_concept_id', None)
        subject_visits = data[subject_id]['visits']

        if not subject_visits or hadm_id not in subject_visits[-1]:
            subject_visits.append([])
        
        if icd9_code:
            subject_visits[-1].append(icd9_code)
        else:
            hadms_with_empty_diagnoses.append(hadm_id)
        
        if drug_code:
            subject_visits[-1].append(f'{drug_code}')
        else:
            hadms_with_empty_drugs.append(hadm_id)

    data = dict(data)

    print(f'hadms_with_empty_diagnoses: {hadms_with_empty_diagnoses}')
    all_codes = sorted(set(c for patient in data.values() for visit in patient['visits'] for c in visit))
    np.random.shuffle(all_codes)

    code_to_index = {code: idx for idx, code in enumerate(all_codes)}
    index_to_code = {idx: code for code, idx in code_to_index.items()}
    vocab_size = len(code_to_index)
    print(f"VOCAB SIZE: {vocab_size}")

    print("Adding labels")
    with open(f"{current_directory}/hcup_ccs_2015_definitions_benchmark.yaml") as definitions_file:
        definitions = yaml.safe_load(definitions_file)

    code_to_group = {code: group for group, details in definitions.items() if details['use_in_benchmark'] for code in details['codes']}
    id_to_group = {idx: group for idx, group in enumerate(sorted([k for k in definitions if definitions[k]['use_in_benchmark']]))}
    group_to_id = {group: idx for idx, group in id_to_group.items()}

    for patient in data.values():
        label_vector = np.zeros(len(group_to_id))
        for visit in patient['visits']:
            for code in visit:
                if code in code_to_group:
                    label_vector[group_to_id[code_to_group[code]]] = 1
        patient['labels'] = label_vector

    print("Converting visits to indices")
    for patient in data.values():
        patient['visits'] = [[code_to_index[code] for code in visit] for visit in patient['visits']]

    num_visits = [len(patient['visits']) for patient in data.values()]
    visit_lengths = [len(visit) for patient in data.values() for visit in patient['visits']]
    print(f"MAX VISITS PER RECORD: {max(num_visits)}, AVG VISITS PER RECORD: {np.mean(num_visits)}, MIN VISITS PER RECORD: {min(num_visits)}")
    print(f"MAX ICD9 CODES PER VISIT: {max(visit_lengths)}, AVG CODES PER VISIT: {np.mean(visit_lengths)}, MIN ICD9 CODES PER VISIT: {min(visit_lengths)}")
    print(f"EMPTY VISITS: {sum(1 for visit_length in visit_lengths if visit_length == 0)}")
    print(f"TOTAL RECORDS: {len(data)}, LONGITUDINAL RECORDS: {sum(1 for subj_info in data.values() if len(subj_info['visits']) > 1)}")

    print("Splitting datasets")
    data_list = list(data.values())
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=SEED, shuffle=True)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=SEED, shuffle=True)

    print("Saving datasets and mappings")
    pickle.dump(train_data, open(f"{current_directory}/train.pkl", "wb"))
    pickle.dump(val_data, open(f"{current_directory}/val.pkl", "wb"))
    pickle.dump(test_data, open(f"{current_directory}/test.pkl", "wb"))

    with open(f"{current_directory}/idToLabel.json", "w") as file:
        json.dump(id_to_group, file)

    meta = {
        'vocab_size': vocab_size,
        'itos': index_to_code,
        'stoi': code_to_index,
    }
    with open(os.path.join(current_directory, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    with open(os.path.join(current_directory, "vocab_stoi.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["String", "Token Index"])
        for token, index in code_to_index.items():
            writer.writerow([token, index])

    print("Done!")