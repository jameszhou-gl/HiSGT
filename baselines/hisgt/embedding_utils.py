import numpy as np
import torch
import json
import pickle
import pandas as pd
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel


def create_label_special_token_dict(dataset, dataset_version, code_vocab_size, label_vocab_size):
    """
    Create an OrderedDict for label and special tokens, ensuring order is preserved.
    """
    # Define labels and special tokens
    label_special_tokens_dict = OrderedDict()
    with open(f'data/{dataset}/{dataset_version}/idToLabel.json', "r") as f:
        idToLabel = json.load(f)

    # Add labels first
    for idx, label in idToLabel.items():
        label_special_tokens_dict[f"LABEL_{idx}"] = {
            "index": code_vocab_size + int(idx),
            "semantic_meaning": f"Chronic condition: {label}. This label describes a specific condition associated with the patient."
        }

    # Add special tokens in a defined order
    label_tokens_dict = OrderedDict({
        "START_RECORD": {"index": code_vocab_size + label_vocab_size, "semantic_meaning": "This token indicates the beginning of a patient's record."},
        "END_LABEL": {"index": code_vocab_size + label_vocab_size + 1, "semantic_meaning": "This token signifies the ending of disease conditions."},
        "END_VISIT": {"index": code_vocab_size + label_vocab_size + 2, "semantic_meaning": "This token denotes the conclusion of a specific medical visit."},
        "END_RECORD": {"index": code_vocab_size + label_vocab_size + 3, "semantic_meaning": "This token marks the end of a patient's record."},
        "PADDING": {"index": code_vocab_size + label_vocab_size + 4, "semantic_meaning": "This token is used for padding sequences."}
    })

    # Update label_special_tokens_dict with special tokens
    label_special_tokens_dict.update(label_tokens_dict)
    return label_special_tokens_dict


def load_semantic_embeddings(dataset, dataset_version, code_vocab_size, column_name="SHORT_TITLE"):
    """Load ICD code embeddings and ensure consistency with the dataset."""
    if dataset=='mimiciii' and dataset_version=='1.4_convert_icd10':
        vocab_semantic_embed_path = f'data/{dataset}/{dataset_version}/icd10_part/vocab_semantic_embed.npz'
    elif dataset == 'mimiciv' and dataset_version == '2.2_icd9_subset_convert_icd10':
        vocab_semantic_embed_path = f'data/{dataset}/{dataset_version}/icd10_part/vocab_semantic_embed.npz'
    else:
        vocab_semantic_embed_path =  f"data/{dataset}/{dataset_version}/vocab_semantic_embed.npz"
    complete_embedding_matrix = np.load(vocab_semantic_embed_path)['embedding_arr']
    complete_icd_codes = np.load(vocab_semantic_embed_path)['icd9_code']
    with open(f"data/{dataset}/{dataset_version}/meta.pkl", 'rb') as f:
        meta = pickle.load(f)
    icd_codes = list(meta['stoi'].keys())[:code_vocab_size]

    # Filter embeddings to include only used codes
    code_embeddings_dict = {code: embedding for code, embedding in zip(
        complete_icd_codes, complete_embedding_matrix) if code in icd_codes}
    icd_embeddings = np.array(
        [code_embeddings_dict[code] for code in icd_codes])

    assert len(icd_embeddings) == len(
        icd_codes), "Filtered embedding matrix size mismatch!"
    return icd_embeddings, icd_codes


def load_hierarchy_embeddings(dataset, dataset_version, code_vocab_size, total_vocab_size, device="cpu"):
    # Load hierarchical embeddings and node-to-ID mapping
    if dataset == 'mimiciii' and dataset_version == '1.4_convert_icd10':
        embedding_file = f'data/{dataset}/{dataset_version}/icd10_part/complete_icd10_hierarchy_embed.npz'
        vocab_file = f'data/{dataset}/{dataset_version}/icd10_part/vocab_stoi_icd10.csv'
    elif dataset == 'mimiciv' and dataset_version == '2.2_icd9_subset_convert_icd10':
        embedding_file = f'data/{dataset}/{dataset_version}/icd10_part/complete_icd10_hierarchy_embed.npz'
        vocab_file = f'data/{dataset}/{dataset_version}/icd10_part/vocab_stoi_icd10.csv'
    else:
        # todo
        # raise NotImplementedError("Hierarchy embeddings not implemented for this dataset")
        return None, None
    data = np.load(embedding_file, allow_pickle=True)
    hierarchy_embeddings = data["embeddings"]
    node_to_id = dict(data["node_to_id"])
    node_to_id = {k: int(v) for k, v in node_to_id.items()}
    vocab_df = pd.read_csv(vocab_file)
    # Initialize a list to store results
    hierarchy_vector_list = []

    # Iterate through each row in the vocabulary
    for _, row in vocab_df.iterrows():
        hierarchy_code = row["ICD-10-Align-Hierarchy"]
        if hierarchy_code in node_to_id:
            node_id = node_to_id[hierarchy_code]
            hierarchy_vector = hierarchy_embeddings[node_id]
            hierarchy_vector_list.append(hierarchy_vector)
        else:
            print(f"Warning: {hierarchy_code} not found in node_to_id mapping.")
            hierarchy_vector_list.append(np.nan)  # Handle missing codes

    # (6984, 384) for mimiciii 1.4_convert_icd10
    hierarchy_vectors = np.array(hierarchy_vector_list)
    assert hierarchy_vectors.shape[0] == code_vocab_size, f"Expected {code_vocab_size} hierarchy vectors, got {hierarchy_vectors.shape[0]}"
    embedding_dim = hierarchy_vectors.shape[1]
    label_special_token_hierarchy_vectors = np.zeros((total_vocab_size - code_vocab_size, embedding_dim))
    total_hierarchy_vectors = np.vstack((hierarchy_vectors, label_special_token_hierarchy_vectors))
    return torch.tensor(total_hierarchy_vectors, dtype=torch.float32).to(device), embedding_dim
    

def create_label_special_token_semantic_embeddings(label_special_tokens_dict):
    """Generate semantic embeddings for special tokens."""
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    semantic_model = AutoModel.from_pretrained(model_name)
    semantic_model.eval()
    label_special_embeddings = []
    label_special_token_embeddings_dict = OrderedDict()

    with torch.no_grad():
        for token_name, token_info in label_special_tokens_dict.items():
            description = token_info["semantic_meaning"]
            inputs = tokenizer(description, return_tensors="pt",
                               padding=True, truncation=True)
            outputs = semantic_model(**inputs)
            # Use mean pooling to generate the embedding
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            label_special_token_embeddings_dict[token_name] = embedding.cpu(
            ).numpy()
            label_special_embeddings.append(embedding.cpu().numpy())
    # todo: check the token_name order in the label_special_embeddings
    return np.array(label_special_embeddings), label_special_token_embeddings_dict


def prepare_total_vocab_semantic_embeddings(icd_embeddings, label_special_embeddings, total_vocab_size, device="cpu"):
    """Concatenate ICD embeddings and special token embeddings."""
    assert icd_embeddings.shape[1] == label_special_embeddings.shape[1], \
        "ICD and special token embeddings dimension mismatch!"
    combined_embeddings = np.vstack(
        (icd_embeddings, label_special_embeddings))
    assert combined_embeddings.shape[0] == total_vocab_size, \
        f"Combined vocab embeddings size mismatch: expected {total_vocab_size}, got {combined_embeddings.shape[0]}"
    return torch.tensor(combined_embeddings, dtype=torch.float32).to(device), combined_embeddings.shape[1]


def validate_vocab_embeddings(vocab_embeddings, icd_codes, label_special_tokens_dict, icd_embeddings, label_special_embeddings):
    """Ensure that vocab_embeddings align with ICD codes and special tokens."""
    # Check ICD code embeddings
    for idx, code in enumerate(icd_codes):
        original_embedding = icd_embeddings[idx]
        vocab_embedding = vocab_embeddings[idx].cpu().numpy()
        assert np.allclose(original_embedding, vocab_embedding), \
            f"Mismatch for ICD code {code} at index {idx}"

    # Check special token embeddings
    label_special_start_idx = len(icd_codes)
    for idx, (token_name, token_info) in enumerate(label_special_tokens_dict.items()):
        label_special_idx = label_special_start_idx + idx
        original_embedding = label_special_embeddings[idx]
        vocab_embedding = vocab_embeddings[label_special_idx].cpu().numpy()
        assert np.allclose(original_embedding, vocab_embedding), \
            f"Mismatch for special token {token_name} at index {label_special_idx}"

    print("Vocab embedding validation passed: All embeddings are consistent!")
    print(f'loaded {len(icd_codes)} ICD embeddings, and {len(label_special_tokens_dict)} label and special token embeddings with dimension {icd_embeddings.shape[1]}')
