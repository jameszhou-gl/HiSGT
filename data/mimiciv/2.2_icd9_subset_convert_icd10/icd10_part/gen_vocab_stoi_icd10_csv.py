from tqdm import tqdm
import time
import csv
import json
import pandas as pd

# Step 1: Convert ICD-9 to ICD-10
# File paths
vocab_file = 'data/mimiciv/2.2_icd9_subset_convert_icd10/vocab_stoi.csv'
mapping_file = 'data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/icd_cm_9_to_10_mapping_from_ethos.csv'
vocab_stoi_icd10_csv_path = 'data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/vocab_stoi_icd10.csv'

print("Loading files...")
vocab_df = pd.read_csv(vocab_file)  # ICD-9 vocabulary
mapping_df = pd.read_csv(mapping_file)  # ICD-9 to ICD-10 mapping

icd9_to_icd10 = dict(zip(mapping_df['icd_9'], mapping_df['icd_10']))

print("Mapping ICD-9 codes to ICD-10...")
mapped_results = []
unmapped_icd9 = []

for _, row in vocab_df.iterrows():
    icd9_code = row['String']  # Extract ICD-9 code
    token_index = row['Token Index']  # Extract token index
    icd10_code = icd9_to_icd10.get(icd9_code, None)  # Map to ICD-10

    if icd10_code:  # If mapping exists
        mapped_results.append(
            {'ICD-9': icd9_code, 'ICD-10': icd10_code, 'Token Index': token_index})
    else:  # If no mapping exists
        unmapped_icd9.append({'ICD-9': icd9_code, 'Token Index': token_index})
        mapped_results.append(
            {'ICD-9': icd9_code, 'ICD-10': 'UNKNOWN', 'Token Index': token_index})

unmapped_count = len(unmapped_icd9)
total_count = len(vocab_df)
print(f"Total ICD-9 codes: {total_count}")

# Step 2: Validate ICD-10 Codes Against Hierarchical Graph
mapped_results_df = pd.DataFrame(mapped_results)
assert mapped_results_df['Token Index'].unique().shape[0] == total_count
mapped_results_df.to_csv(vocab_stoi_icd10_csv_path, index=False)
print("Loading ICD10 hierarchy file...")
with open('data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/icd10_hierarchical_graph.json', 'r') as file:
    icd10_hierarchical_graph = json.load(file)


def normalize_icd10_code(icd10_code, icd10_hierarchy):
    """Normalize ICD-10 code by truncating or finding the nearest match."""
    if icd10_code in icd10_hierarchy:
        return icd10_code  # Code already exists
    if '.' not in icd10_code:
        icd10_code = icd10_code[:3] + '.' + icd10_code[3:]
    # Truncate trailing characters (e.g., "S065X9A" -> "S06.5X")
    flag = False
    search_code = None
    for length in range(3, len(icd10_code)+1):
        if length == 4:
            # Skip the fourth character, e.g., 'S06.'
            continue
        truncated_code = icd10_code[:length]
        if truncated_code in icd10_hierarchy:
            search_code = truncated_code
            flag = True
        else:
            if flag:
                break
    # Return None if no match is found
    return search_code


mapped_results_df['ICD-10-Align'] = mapped_results_df['ICD-10'].apply(
    lambda x: normalize_icd10_code(x, icd10_hierarchical_graph))
mapped_results_df = mapped_results_df[[
    'Token Index', 'ICD-9', 'ICD-10', 'ICD-10-Align']]
mapped_results_df.to_csv(vocab_stoi_icd10_csv_path, index=False)
extra_check_num = mapped_results_df['ICD-10-Align'].isna().sum()
print(f'{extra_check_num} codes need to be extra checked.')


# Sort the DataFrame, placing rows with NaN in 'ICD-10-Align' at the bottom
sort_mapped_results_df = mapped_results_df.sort_values(
    by='ICD-10-Align',
    key=lambda col: col.isna(),
    ascending=True
)
sort_mapped_results_df.to_csv(vocab_stoi_icd10_csv_path, index=False)

# Load supplementary mapping data
supplementary_mapping_df = pd.read_csv(
    'data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/icd_cm_9_to_10_mapping_supplementary.csv')

# Iterate over rows of the mapped results DataFrame
for index, row in sort_mapped_results_df.iterrows():
    if pd.isna(row['ICD-10-Align']):  # Check if ICD-10-Align is NaN
        # Find corresponding ICD-10 code for the ICD-9 code
        supp_icd_10_code = supplementary_mapping_df[supplementary_mapping_df['icd_9']
                                                    == row['ICD-9']]['icd_10'].values
        # Check if corresponding ICD-10 code was found
        if len(supp_icd_10_code) == 0:
            raise ValueError(
                f'No ICD-10 code found for ICD-9 code: {row["ICD-9"]}')

        # Assign the found ICD-10 code to the row
        sort_mapped_results_df.loc[index, 'ICD-10'] = supp_icd_10_code[0]
del sort_mapped_results_df['ICD-10-Align']
sort_mapped_results_df['ICD-10-Align-Hierarchy'] = sort_mapped_results_df['ICD-10'].apply(
    lambda x: normalize_icd10_code(x, icd10_hierarchical_graph))
# sort_mapped_results_df = sort_mapped_results_df.rename(columns={'ICD-10': 'ICD-10-Align-Hierarchy'})
final_mapped_results_df = sort_mapped_results_df
final_mapped_results_df.sort_values(by='Token Index', inplace=True)
final_mapped_results_df = final_mapped_results_df[[
    'Token Index', 'ICD-9', 'ICD-10', 'ICD-10-Align-Hierarchy']]


original_vocab_df = pd.read_csv(vocab_file)  # ICD-9 vocabulary
# Rename to align with the column name in final results
original_vocab_df.rename(columns={'String': 'ICD-9'}, inplace=True)

# Merge the two DataFrames on 'Token Index' for comparison
merged_df = pd.merge(original_vocab_df, final_mapped_results_df,
                     on='Token Index', suffixes=('_original', '_mapped'))

# Check for mismatches in the ICD-9 codes
mismatched_icd9 = merged_df[merged_df['ICD-9_original']
                            != merged_df['ICD-9_mapped']]

# Output results
if mismatched_icd9.empty:
    print("All ICD-9 codes and Token Index values match!")
    final_mapped_results_df.to_csv(vocab_stoi_icd10_csv_path, index=False)
else:
    print("Mismatches found:")
    print(mismatched_icd9)

if final_mapped_results_df['ICD-10-Align-Hierarchy'].isna().sum() != 0:
    temp_num = final_mapped_results_df['ICD-10-Align-Hierarchy'].isna().sum()
    print(f'There are still {temp_num} missing ICD-10 codes.')
    print(
        final_mapped_results_df[final_mapped_results_df['ICD-10-Align-Hierarchy'].isna()]['ICD-9'].to_list())
else:
    print('All ICD-10 codes are successfully mapped.')

if final_mapped_results_df['ICD-10'].unique().shape[0] != original_vocab_df.shape[0]:
    num_unique_icd10 = final_mapped_results_df['ICD-10'].unique().shape[0]
    print(
        f'{original_vocab_df.shape[0]-num_unique_icd10} ICD-10 codes are duplicated.')
    assert original_vocab_df['ICD-9'].unique(
    ).shape[0] == original_vocab_df.shape[0]
    print('But orginal ICD-9 codes are unique, so the model can be trained.')
    print('We use ICD-9 codes (Toekn Index) for the code sequence and ICD-10 codes for semantic consistency to train the model.')

# Step 3: Text Descriptions

# Load hierarchical graph
with open("data/mimiciv/2.2_icd9_subset_convert_icd10/icd10_part/icd10_hierarchical_graph.json", "r") as f:
    hierarchical_graph = json.load(f)

# Function to get SHORT-TITLE from hierarchical graph


def get_short_title(icd10_code):
    node = hierarchical_graph.get(icd10_code)
    return node["description"] if node else "Description not available."


# Load the CSV as a DataFrame
df = pd.read_csv(vocab_stoi_icd10_csv_path)

# Add or update the SHORT-TITLE and LONG-TITLE columns
if "SHORT-TITLE" not in df.columns:
    df["SHORT-TITLE"] = ""
# if "LONG-TITLE" not in df.columns:
#     df["LONG-TITLE"] = ""

# Process each row with tqdm for progress tracking
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    icd10_code = row["ICD-10-Align-Hierarchy"]

    # Get short title
    short_title = get_short_title(icd10_code)
    df.at[index, "SHORT-TITLE"] = short_title

    # # ! comment out to aovid generating long title
    # long_title = generate_long_title(icd10_code, short_title)
    # df.at[index, "LONG-TITLE"] = long_title


# Save the updated DataFrame back to the CSV
df.to_csv(vocab_stoi_icd10_csv_path, index=False)
print(f"Updated CSV saved to {vocab_stoi_icd10_csv_path}")
