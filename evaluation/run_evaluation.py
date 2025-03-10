import warnings
import os
import sys
import random
import torch
import pickle
import json
import argparse
import numpy as np
from collections import defaultdict
from fidelity_metrics_from_halo import fidelity_from_halo_evaluation
from fidelity import fidelity_evaluation
from utility import utility_evaluation
from utility_from_halo import utility_from_halo_evaluation
from privacy import privacy_evaluation
from util import transform_data_matrix

import pdb
# pdb.set_trace()
warnings.filterwarnings('ignore')


class CustomNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap any module that starts with "numpy._core" to "numpy.core"
        if module.startswith("numpy._core"):
            new_module = "numpy.core" + module[len("numpy._core"):]
            return super().find_class(new_module, name)
        return super().find_class(module, name)
# Convert NumPy types to standard Python types before dumping


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert NumPy floats to Python floats
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert NumPy integers to Python integers
    return obj  # Return as-is if not a NumPy type


# Global function to set random seed
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True,
                        type=str, help='Path dir to the synthetic dataset')
    parser.add_argument('--dataset', type=str,
                        default='mimiciii', help="Dataset name")
    parser.add_argument('--dataset_version', type=str,
                        default='1.4', help="Dataset version")
    parser.add_argument('--fidelity_from_halo', action="store_false",
                        help='Whether to evaluate fidelity metrics from HALO')
    parser.add_argument('--fidelity', action="store_false",
                        help='Whether to evaluate fidelity')
    parser.add_argument('--utility', action="store_false",
                        help='Whether to evaluate utility')
    parser.add_argument('--privacy', action="store_false",
                        help='Whether to evaluate privacy')

    # Parse arguments
    args = parser.parse_args()
    print(args)
    if args.dataset == 'mimiciii' and args.dataset_version.startswith('1.4'):
        real_data_path = f"data/mimiciii/{args.dataset_version}/train.pkl"
        test_data_path = f"data/mimiciii/{args.dataset_version}/test.pkl"
        meta_data_path = f"data/mimiciii/{args.dataset_version}/meta.pkl"
        if not os.path.exists(real_data_path) or not os.path.exists(meta_data_path):
            raise FileNotFoundError("Real dataset or metadata file not found.")

        with open(real_data_path, 'rb') as f:
            real_data = pickle.load(f)
        
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)

        with open(meta_data_path, 'rb') as f:
            meta = pickle.load(f)
            try:
                code_vocab_size = meta['vocab_size'] - len(meta['special_tokens'])
            except:
                code_vocab_size = meta['vocab_size']
    elif args.dataset == 'mimiciv' and args.dataset_version.startswith('2.2'):
        real_data_path = f"data/mimiciv/{args.dataset_version}/train.pkl"
        test_data_path = f"data/mimiciv/{args.dataset_version}/test.pkl"
        meta_data_path = f"data/mimiciv/{args.dataset_version}/meta.pkl"
        if not os.path.exists(real_data_path) or not os.path.exists(meta_data_path):
            raise FileNotFoundError("Real dataset or metadata file not found.")

        with open(real_data_path, 'rb') as f:
            real_data = pickle.load(f)

        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)

        with open(meta_data_path, 'rb') as f:
            meta = pickle.load(f)
            code_vocab_size = meta['vocab_size']
    else:
        raise ValueError("Unsupported dataset or dataset version specified.")

    # Load synthetic data
    synthetic_files = [file for file in os.listdir(
        args.output_path) if file.endswith('.pkl')]
    if not synthetic_files:
        raise FileNotFoundError(
            "No synthetic dataset file found in the specified output path.")
    
    for synthetic_file in synthetic_files:
        synthetic_file_path = os.path.join(args.output_path, synthetic_file)
        print(f"Found synthetic file: {synthetic_file_path}")
        evaluation_file = synthetic_file_path.replace(
                '.pkl', '_evaluation_metrics.json')
        with open(synthetic_file_path, 'rb') as f:
            # synthetic_data = pickle.load(f)
            synthetic_data = CustomNumpyUnpickler(f).load()

        # Prepare matrices for evaluation
        real_data_matrices = {
            matrix_type: transform_data_matrix(
                real_data, code_vocab_size, matrix_type)
            for matrix_type in ["binary", "count", "probability"]
        }
        synthetic_data_matrices = {
            matrix_type: transform_data_matrix(
                synthetic_data, code_vocab_size, matrix_type)
            for matrix_type in ["binary", "count", "probability"]
        }

        # Align number of rows
        if real_data_matrices["binary"].shape[0] != synthetic_data_matrices["binary"].shape[0]:
            min_rows = min(
                real_data_matrices["binary"].shape[0], synthetic_data_matrices["binary"].shape[0])
            print(f"Truncating to {min_rows} rows")
            for matrix_type in real_data_matrices.keys():
                real_data_matrices[matrix_type] = real_data_matrices[matrix_type][:min_rows]
                synthetic_data_matrices[matrix_type] = synthetic_data_matrices[matrix_type][:min_rows]

        # Initialize results dictionary
        results = defaultdict(lambda: defaultdict(dict))
        if args.fidelity_from_halo:
            print("Evaluating fidelity metrics from HALO...")
            results["fidelity_from_halo"] = fidelity_from_halo_evaluation(
                real_data, synthetic_data, args.output_path)

        if args.fidelity:
            print("Evaluating fidelity...")
            for matrix_type in ["binary", "count", "probability"]:
                print(f"Evaluating with {matrix_type} matrix...")
                fidelity_results = fidelity_evaluation(
                    real_data_matrices[matrix_type],
                    synthetic_data_matrices[matrix_type],
                    matrix_type
                )
                results['fidelity'].update(fidelity_results)
            # Save the results as a JSON file
            with open(evaluation_file, 'w') as f:
                json.dump(results, f, indent=4, default=convert_numpy)
        if args.utility:
            if (args.dataset == 'mimiciii' and args.dataset_version.startswith('1.4')) or (args.dataset == 'mimiciv' and args.dataset_version.startswith('2.2')):
                print("Evaluating utility...")
                results['utility_from_halo'] = utility_from_halo_evaluation(code_vocab_size,
                    args, synthetic_data)
            else:
                # Placeholder for other dataset types
                print(f"Utility evaluation for {args.dataset}_{args.dataset_version} is not implemented yet.")
            # Save the results as a JSON file
            with open(evaluation_file, 'w') as f:
                json.dump(results, f, indent=4, default=convert_numpy)
        if args.privacy:
            print("Evaluating privacy...")
            results["privacy"] = privacy_evaluation(real_data, test_data, synthetic_data)
            # Save the results as a JSON file
            with open(evaluation_file, 'w') as f:
                json.dump(results, f, indent=4, default=convert_numpy)
        print(f"Saved evaluation results to: {evaluation_file}")


if __name__ == "__main__":
    main()
