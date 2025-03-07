### Notes for Processing MIMIC-IV
> You only need to perform step 1 and 2 for a quick start.

1. **Prepare Raw Data:**  
   Copy `admissions.csv` and `diagnoses_icd.csv` from the downloaded **MIMIC-IV v2.2** dataset into `data/mimiciv/2.2_icd9_subset_convert_icd10/`.

2. **Build Dataset:**  
   Run `build_dataset.py` to generate:  
   - `train.pkl, val.pkl, test.pkl` (data splits)  
   - `idToLabel.json` (disease label mapping)  
   - `vocab_stoi.csv` (vocabulary)  
   - `meta.pkl` (metadata)  

3. **ICD-9 to ICD-10 Conversion:**  
   - Since the **hierarchical structure** is based on ICD-10 (`icd10_hierarchical_graph.json`) and **disease labels** use ICD-9, we choose the ICD-9 subset in MIMIC-IV to obtain the disease labels for the utility evaluation. But then we convert ICD-9 to ICD-10 to get the hierarchical embeddings.  
   - `gen_vocab_stoi_icd10_csv.py` would convert ICD-9 in vocabulary to ICD-10 codes.  
   - `gen_vocab_icd10_semantic_embed.py` would generate semantic embeddings for each ICD-10 code in the vocabulary.  
   - `gen_vocab_icd10_semantic_embed.py` would generate semantic embeddings for each ICD-10 code in the vocabulary.  
 > You don't need perform step 3 since hierarchical embeddings and semantic embeddings are already provided.

4. **Embedding Files:**  
   - **Semantic embeddings:** `vocab_semantic_embed.npz`  
   - **Hierarchical embeddings:** `complete_icd10_hierarchy_embed.npz`  
   Both are stored in `data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part`.  
  

