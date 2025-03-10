from sklearn.metrics import r2_score
import numpy as np


SEED = 1337
np.random.seed(SEED)


def compute_prevalence(data_matrix, matrix_type):
    """
    Computes prevalence for the given matrix type.

    Args:
        data_matrix: Transformed data matrix (binary, count, or probability).
        matrix_type: Type of matrix ('binary', 'count', or 'probability').

    Returns:
        numpy.ndarray: Prevalence values for each code.
    """
    if matrix_type == "binary":
        # Proportion of patients with each code
        return np.mean(data_matrix > 0, axis=0)
    elif matrix_type in ["count", "probability"]:
        # Mean frequency or normalized probabilities
        return np.mean(data_matrix, axis=0)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")


def fidelity_evaluation(real_data, synthetic_data, matrix_type):
    """
    Evaluates fidelity metrics using the specified matrix type.

    Args:
        real_data: Real dataset matrix.
        synthetic_data: Synthetic dataset matrix.
        matrix_type: The type of matrix used ('binary', 'count', or 'probability').

    Returns:
        dict: Fidelity metrics.
    """
    results = {}

    # Prevalence-based metrics
    real_prevalence = compute_prevalence(real_data, matrix_type)
    synthetic_prevalence = compute_prevalence(synthetic_data, matrix_type)
    results[f"{matrix_type}_mmd"] = np.abs(
        real_prevalence - synthetic_prevalence).max()
    # R² score
    results[f"{matrix_type}_r2_score"] = r2_score(
        real_prevalence, synthetic_prevalence)
    

    return results

