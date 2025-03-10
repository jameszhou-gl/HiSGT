import torch
import csv
import random
import argparse
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

# Arguments
parser = argparse.ArgumentParser(
    description="Generate semantic embeddings for ICD-10 codes in vocabulary.")
parser.add_argument("--model_name", type=str,
                    default="emilyalsentzer/Bio_ClinicalBERT", help="Pretrained model name.")
parser.add_argument("--description_type", type=str, default="SHORT-TITLE",
                    choices=['SHORT-TITLE', 'LONG-TITLE'], help="Column to use for descriptions.")
args = parser.parse_args()

# Main Execution
if __name__ == "__main__":
    # Load descriptions
    csv_path = "data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/vocab_stoi_icd10.csv"
    output_path = "data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/vocab_semantic_embed"
    # token_to_description = {}
    token_list, icd9_list, icd10_list, icd10_align_hierarchy_list, description_list = [], [], [], [], []
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            token_index = row["Token Index"]
            icd9_code = row["ICD-9"]
            icd10_code = row["ICD-10"]
            icd10_code_align = row["ICD-10-Align-Hierarchy"]
            description = row.get(args.description_type, "")
            if icd10_code_align and description:
                token_list.append(token_index)
                icd9_list.append(icd9_code)
                icd10_list.append(icd10_code)
                icd10_align_hierarchy_list.append(icd10_code_align)
                description_list.append(description)

    # Generate embeddings
    # Load the tokenizer and model
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embedding_list = []
    with torch.no_grad():
        for idx in tqdm(range(len(token_list)), desc="Generating Embeddings"):
            # Tokenize the description
            inputs = tokenizer(description_list[idx], return_tensors="pt",
                               padding=True, truncation=True)
            # Get the embeddings (mean pooling for simplicity)
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            embedding_list.append(embeddings.cpu().numpy())

    # Save embeddings
    # Convert dictionary to a NumPy array for saving
    embedding_arr = np.array(embedding_list)
    np.savez_compressed(output_path, token_index=token_list, icd9_code=icd9_list,
                        icd10_code=icd10_list, icd10_align=icd10_align_hierarchy_list, embedding_arr=embedding_arr)
    print(f"Embeddings saved to {output_path}")
