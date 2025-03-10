# Instructions for running the SynTEG baseline

Breaking it down into the four main steps

## Step 1: Run dependency learning
```
python3 dependency_learning.py
```

This will:
- Load training and validation datasets
- Train the dependency model for 50 epochs
- Save the best model to `../../save/synteg_dependency_model`

## Step 2: Export condition vectors
```
python3 export_condition.py
```

This will:
- Load the trained dependency model
- Process the training dataset
- Generate condition vectors
- Save the condition dataset to `data/conditionDataset.pkl`

## Step 3: Run condition simulation
```
python3 condition_simulation.py
```

This will:
- Train the Generator and Discriminator for 600 epochs
- Save the model every 50 epochs
- Save the final model to `../../save/synteg_condition_model`

## Step 4: Create temporary directory for synthetic data
```
mkdir -p ../../temp_synteg
```

## Step 5: Generate Data
```
python3 generate_data.py
```

This will:
- Load both trained models (dependency and condition)
- Generate synthetic EHR data
- Save synthetic data in batches to `../../temp_synteg/syntegDataset_{count}.pkl`