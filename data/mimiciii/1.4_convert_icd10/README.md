### Notes for Processing MIMIC-III  
> You only need to perform step 1 and 2 for a quick start.


1. **Prepare Raw Data:**  
   Copy `ADMISSIONS.csv` and `DIAGNOSES_ICD.csv` from the downloaded **MIMIC-III v1.4** dataset into `data/mimiciii/1.4_convert_icd10/`.  

2. **Build Dataset:**  
   Run `build_dataset.py` to generate:  
   - `train.pkl, val.pkl, test.pkl` (data splits)  
   - `idToLabel.json` (disease label mapping)  
   - `vocab_stoi.csv` (vocabulary)  
   - `meta.pkl` (metadata)  

3. **ICD-9 to ICD-10 Conversion:**  
   - The **hierarchical structure** is based on ICD-10 (`icd10_hierarchical_graph.json`), but **disease labels** use ICD-9.  
   - Convert ICD-9 codes in the dataset to ICD-10 with `process.ipynb` (You don't need to run it because hierarchical embeddings and semantic embeddings are already provided in the folder.).  
   - This notebook also handles **semantic** and **hierarchical** embeddings.  

4. **Embedding Files:**  
   - **Semantic embeddings:** `vocab_semantic_embed.npz`  
   - **Hierarchical embeddings:** `complete_icd10_hierarchy_embed.npz`  
   Both are stored in `data/mimiciii/1.4_convert_icd10/icd10_part/`.  