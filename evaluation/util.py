import numpy as np


def data_to_binary_matrix(data, code_vocab_size):
    """
    Converts data to a binary matrix.
    Each row represents a patient, each column represents a unique code, and a 1 indicates the presence of the code.
    """
    binary_matrix = np.zeros((len(data), code_vocab_size))
    for i, patient in enumerate(data):
        unique_codes = set(
            code for visit in patient['visits'] for code in visit)
        for code in unique_codes:
            binary_matrix[i, code] = 1
    return binary_matrix


def data_to_count_matrix(data, code_vocab_size):
    """
    Converts data to a count matrix.
    Each row represents a patient, each column represents a unique code, and the value indicates the count of the code.
    """
    count_matrix = np.zeros((len(data), code_vocab_size), dtype=int)
    for i, patient in enumerate(data):
        for visit in patient['visits']:
            for code in visit:
                count_matrix[i, code] += 1
    return count_matrix


def transform_data_matrix(data, code_vocab_size, matrix_type):
    """
    Transforms data into binary, count, or probability matrix.

    Args:
        data: Input data with the shape of [num_patients, code_vocab_size].
        code_vocab_size: Vocabulary size of codes.
        matrix_type: Type of matrix ('binary', 'count', 'probability').

    Returns:
        numpy.ndarray: Transformed matrix with the shape of [num_patients, code_vocab_size].
    """
    if matrix_type == "binary":
        return data_to_binary_matrix(data, code_vocab_size)
    elif matrix_type == "count":
        return data_to_count_matrix(data, code_vocab_size)
    elif matrix_type == "probability":
        count_matrix = data_to_count_matrix(data, code_vocab_size)
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        return count_matrix / (row_sums + 1e-5)  # Avoid division by zero
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")
