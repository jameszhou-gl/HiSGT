import pandas as pd
import os
from sqlalchemy import create_engine

def OMOP_to_ICD9_conversion(args=None, db_name="postgres", db_config=None,save_csv=True):

    db_connection = db_config["db"]

    if db_name=='csv':
        # CSV Mode
        omop_vocabs_path = f"data/{args.dataset}/{args.dataset_version}/"+db_config["csv"]["omopvocabsfolderpath"]
        input_data_path = f"data/{args.dataset}/{args.dataset_version}/"+db_config["csv"]["inputdatafolderpath"]

        concept_file = os.path.join(omop_vocabs_path, 'CONCEPT.csv')
        concept_relationship_file = os.path.join(omop_vocabs_path, 'CONCEPT_RELATIONSHIP.csv')
        condition_occurrence_file = os.path.join(input_data_path, 'condition_occurrence.csv')
        visit_occurrence_file = os.path.join(input_data_path, 'visit_occurrence.csv')
        query_drug_exposure = os.path.join(input_data_path, 'drug_exposure.csv')

        # Load data from CSV
        df_concept = pd.read_csv(concept_file, sep="\t", dtype=str)
        df_cond_occurence = pd.read_csv(condition_occurrence_file, dtype=str)
        df_concept_relationship = pd.read_csv(concept_relationship_file, sep="\t", dtype=str)
        df_visit_occurrence = pd.read_csv(visit_occurrence_file, dtype=str)
        df_drug_exposure = pd.read_csv(query_drug_exposure, dtype=str)

    elif db_name=='postgres':
        # Database Mode
        try:
            db_engine = create_engine(f"postgresql+psycopg2://{db_connection['user']}:{db_connection['password']}@{db_connection['host']}:{db_connection['port']}/{db_connection['database']}")
            print("Database connection established successfully!")
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

        query_concept = "SELECT * FROM \"CONCEPT\";"
        query_concept_relationship = "SELECT * FROM \"CONCEPT_RELATIONSHIP\";"
        query_condition_occurrence = "SELECT * FROM \"condition_occurrence\";"
        query_visit_occurrence = "SELECT * FROM \"visit_occurrence\";"
        query_drug_exposure = "SELECT * FROM \"drug_exposure\";"

        # Load data from Database
        df_concept = pd.read_sql(query_concept, db_engine)
        df_concept_relationship = pd.read_sql(query_concept_relationship, db_engine)
        df_cond_occurence = pd.read_sql(query_condition_occurrence, db_engine)
        df_visit_occurrence = pd.read_sql(query_visit_occurrence, db_engine)
        df_drug_exposure = pd.read_sql(query_drug_exposure, dtype=str)
    elif db_name=='sql':
        # Database Mode
        try:
            db_engine = create_engine(f"mysql+mysqldb://{db_connection['user']}:{db_connection['password']}@{db_connection['host']}:{db_connection['port']}/{db_connection['database']}")
            print("Database connection established successfully!")
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

        query_concept = "SELECT * FROM concept;"
        query_concept_relationship = "SELECT * FROM concept_relationship;"
        query_condition_occurrence = "SELECT * FROM condition_occurrence;"
        query_visit_occurrence = "SELECT * FROM visit_occurrence;"
        query_drug_exposure = "SELECT * FROM drug_exposure;"

        # Load data from Database
        df_concept = pd.read_sql(query_concept, db_engine)
        df_concept_relationship = pd.read_sql(query_concept_relationship, db_engine)
        df_cond_occurence = pd.read_sql(query_condition_occurrence, db_engine)
        df_visit_occurrence = pd.read_sql(query_visit_occurrence, db_engine)
        df_drug_exposure = pd.read_csv(query_drug_exposure, dtype=str)


    # Step 1: Filter df_concept to only ICD-9 concepts
    icd9_concepts = df_concept[df_concept['vocabulary_id'] == 'ICD9CM']

    # Step 2: Filter df_concept_relationship to only 'Maps to' relationships where concept_id_1 is in ICD-9
    icd9_mappings = df_concept_relationship[
        (df_concept_relationship['relationship_id'] == 'Maps to') &
        (df_concept_relationship['concept_id_1'].isin(icd9_concepts['concept_id']))
    ]

    # Step 3: Merge condition occurrences with standard OMOP concepts
    merged_df = df_cond_occurence.merge(
        df_concept, 
        left_on='condition_concept_id', 
        right_on='concept_id', 
        how='left'
    ).rename(columns={'concept_name': 'omop_condition_name'}).drop(columns=['concept_id'])

    # Step 4: Merge with the filtered ICD-9 concept relationships
    merged_df = merged_df.merge(
        icd9_mappings,  
        left_on='condition_concept_id', 
        right_on='concept_id_2',  
        how='inner'  # Ensures only mapped conditions remain
    )

    # Debug: Check intermediate merge
    print("After merging with relationships:")
    print(merged_df[['condition_concept_id', 'concept_id_1', 'concept_id_2']].head())

    # Step 5: Merge again to get ICD-9 names and codes
    merged_df = merged_df.merge(
        icd9_concepts,  
        left_on='concept_id_1', 
        right_on='concept_id', 
        how='left'
    )

    # Debug: Check if 'concept_code' exists
    print("Columns in df_concept:", df_concept.columns)
    print("Columns in merged_df after final merge:", merged_df.columns)

    # Rename columns
    if 'concept_code' in merged_df.columns:
        merged_df = merged_df.rename(columns={'concept_name': 'icd9_condition_name', 'concept_code': 'icd9_code'})
    else:
        print("Warning: 'concept_code' column missing in df_concept!")

    # Drop unnecessary columns
    merged_df = merged_df.drop(columns=['concept_id', 'concept_id_1', 'concept_id_2'], errors='ignore')

    # Debug: Check final result before dropping NaNs
    print("Final merged dataframe preview:")
    print(merged_df.head())

    # Step 6: Ensure only valid ICD-9 codes remain
    if 'icd9_code' in merged_df.columns:
        merged_df = merged_df.dropna(subset=['icd9_code'])
    else:
        print("Error: 'icd9_code' column is missing, skipping dropna step.")

    # Step 7: Display final results
    # print(merged_df.head())


    # Merge the merged_df with visit_occurrence based on visit_occurrence_id
    merged_df_with_visit = merged_df.merge(
        df_visit_occurrence, 
        on='visit_occurrence_id',  # Merge on visit_occurrence_id
        how='left'  # or 'inner' depending on your requirements
    ).drop(columns=['person_id_y']).rename(columns={'person_id_x': 'person_id', 'concept_code_y': 'ICD9_CODE', 'concept_name_y': 'visit_concept_name'})


        # Optional: Sort by person_id and visit_start_date to ensure correct sequencing
    merged_df_with_visit_sorted = merged_df_with_visit.sort_values(['visit_start_date','person_id'], ascending=[True, True])
    merged_df_with_visit_sorted['ICD9_CODE'] = merged_df_with_visit_sorted['ICD9_CODE'].str.replace('.','')

    merged_df_with_visit_sorted[['visit_occurrence_id','person_id','visit_start_date', 'ICD9_CODE']]


    merged_df_with_visit_clean = merged_df_with_visit_sorted.drop(columns=['provider_id_y','domain_id_y','vocabulary_id_x','standard_concept_y',\
    'invalid_reason_y','valid_start_date_x','valid_start_date_y','valid_end_date_x','valid_end_date_y'])\
    .rename(columns={'provider_id_x': 'provider_id',\
                     'domain_id_x':'domain_id',\
                        'vocabulary_id_y':'vocabulary_id',\
                            'standard_concept_x':'standard_concept',\
                                'concept_code_x':'concept_code',\
                                    'invalid_reason_x':'invalid_reason',\
                                        'concept_class_id_x':'concept_class_id',\
                                            'concept_class_id_y':'concept_class_id_code'})
    
    merged_df_with_visit_clean['visit_start_datetime'] = pd.to_datetime(merged_df_with_visit_clean['visit_start_datetime'])
    merged_df_with_visit_clean = merged_df_with_visit_clean.sort_values('visit_start_datetime').reset_index(drop=True)

    # merge drug_exposure with merged_df_with_visit_clean
    df_drug_exposure = df_drug_exposure.merge(df_concept[['concept_id', 'concept_name']], left_on='drug_concept_id', right_on='concept_id', how='left').rename(columns={'concept_name': 'drug_name'}).drop(columns=['concept_id'])
    merged_df_with_visit_clean = merged_df_with_visit_clean.merge(df_drug_exposure[['person_id', 'drug_concept_id','drug_name', 'drug_exposure_start_date','visit_occurrence_id']], on='visit_occurrence_id', how='left')\
        .drop(columns=['person_id_y']).rename(columns={'person_id_x': 'person_id'})
    
    # df_person = df_person.merge(df_concept[['concept_id', 'concept_name']], left_on='gender_concept_id', right_on='concept_id', how='left').rename(columns={'concept_name': 'gender'}).drop(columns=['concept_id'])
    # df_person = df_person.merge(df_concept[['concept_id', 'concept_name']], left_on='race_concept_id', right_on='concept_id', how='left').rename(columns={'concept_name': 'race'}).drop(columns=['concept_id'])

    
    if save_csv:
        merged_df_with_visit_clean.to_csv('synthea1k_ICD9.csv',index=True)

    return merged_df_with_visit_clean



   