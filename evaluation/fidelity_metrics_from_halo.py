import os
import json
import numpy as np
import pickle
import itertools
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Function to calculate mean lengths
def calculate_mean_lengths(data):
    record_lengths = [len(patient['visits']) for patient in data]
    visit_lengths = [len(visit)
                     for patient in data for visit in patient['visits']]
    return {
        'mean_record_length': np.mean(record_lengths),
        'std_record_length': np.std(record_lengths),
        'mean_visit_length': np.mean(visit_lengths),
        'std_visit_length': np.std(visit_lengths),
    }


# Function to calculate unigram and bigram probabilities
def calculate_code_probabilities(data):
    unigram_counts = {}
    sequential_bigram_counts = {}
    same_visit_bigram_counts = {}
    total_visits = 0

    for patient in data:
        for visit in patient['visits']:
            total_visits += 1
            # Unigrams
            for code in visit:
                unigram_counts[code] = unigram_counts.get(code, 0) + 1

            # Same-visit bigrams
            for bigram in itertools.combinations(visit, 2):
                bigram = tuple(sorted(bigram))
                same_visit_bigram_counts[bigram] = same_visit_bigram_counts.get(
                    bigram, 0) + 1

        # Sequential bigrams
        for i in range(len(patient['visits']) - 1):
            for code1 in patient['visits'][i]:
                for code2 in patient['visits'][i + 1]:
                    bigram = (code1, code2)
                    sequential_bigram_counts[bigram] = sequential_bigram_counts.get(
                        bigram, 0) + 1

    total_unigrams = sum(unigram_counts.values())
    unigram_probs = {code: count / total_unigrams for code,
                     count in unigram_counts.items()}

    total_same_visit_bigrams = sum(same_visit_bigram_counts.values())
    same_visit_bigram_probs = {
        bigram: count / total_same_visit_bigrams for bigram, count in same_visit_bigram_counts.items()}

    total_sequential_bigrams = sum(sequential_bigram_counts.values())
    sequential_bigram_probs = {
        bigram: count / total_sequential_bigrams for bigram, count in sequential_bigram_counts.items()}

    return unigram_probs, same_visit_bigram_probs, sequential_bigram_probs


# Function to compare probabilities
def compare_probabilities(real_probs, synthetic_probs, label, save_dir):
    real_values = []
    synthetic_values = []

    for key in set(real_probs.keys()).union(synthetic_probs.keys()):
        real_values.append(real_probs.get(key, 0))
        synthetic_values.append(synthetic_probs.get(key, 0))

    r2 = r2_score(real_values, synthetic_values)

    # Calculate deviation from the reference line y=x
    deviation = np.abs(np.array(real_values) - np.array(synthetic_values))

    # Normalize the deviation to range [0, 1] for consistent coloring
    deviation_normalized = (deviation - deviation.min()) / \
        (deviation.max() - deviation.min())
    deviation_normalized = 1 - deviation_normalized

    # Create a scatter plot with colormap
    # plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        real_values, synthetic_values, c=deviation_normalized, cmap="viridis", alpha=0.7, s=20, label="Data points"
    )

    # Add diagonal reference line (y=x)
    max_val = max(max(real_values), max(synthetic_values)) * 1.1
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--",
             linewidth=1, label="Reference line (y=x)")

    # Add colorbar for reference
    cbar = plt.colorbar(scatter)
    # cbar.set_label("Gradient Metric (e.g., Density or Order)", fontsize=10)

    # Add gridlines and beautify
    plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xlabel("Real Data", fontsize=12)
    plt.ylabel("Synthetic Data", fontsize=12)
    plt.title(f"{label} Comparison ($R^2 = {r2:.2f}$)",
              fontsize=12)
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save the beautified plot
    plot_path = os.path.join(
        save_dir, f"{label.replace(' ', '_')}_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return r2


def fidelity_from_halo_evaluation(real_data, synthetic_data, save_dir):
    # Generate statistics for real and synthetic data
    real_lengths = calculate_mean_lengths(real_data)
    synthetic_lengths = calculate_mean_lengths(synthetic_data)

    real_unigrams, real_same_bigrams, real_seq_bigrams = calculate_code_probabilities(
        real_data)
    synthetic_unigrams, synthetic_same_bigrams, synthetic_seq_bigrams = calculate_code_probabilities(
        synthetic_data)

    # Compare probabilities and save plots
    r2_unigrams = compare_probabilities(
        real_unigrams, synthetic_unigrams, "Unigram Probabilities", save_dir)
    r2_same_bigrams = compare_probabilities(
        real_same_bigrams, synthetic_same_bigrams, "Same Visit Bigram Probabilities", save_dir)
    r2_seq_bigrams = compare_probabilities(
        real_seq_bigrams, synthetic_seq_bigrams, "Sequential Visit Bigram Probabilities", save_dir)

    # Save metrics to JSON
    metrics = {
        "real_record_mean_length": real_lengths['mean_record_length'],
        "real_record_std_length": real_lengths['std_record_length'],
        "real_visit_mean_length": real_lengths['mean_visit_length'],
        "real_visit_std_length": real_lengths['std_visit_length'],
        "synthetic_record_mean_length": synthetic_lengths['mean_record_length'],
        "synthetic_record_std_length": synthetic_lengths['std_record_length'],
        "synthetic_visit_mean_length": synthetic_lengths['mean_visit_length'],
        "synthetic_visit_std_length": synthetic_lengths['std_visit_length'],
        "r2_unigrams": r2_unigrams,
        "r2_same_bigrams": r2_same_bigrams,
        "r2_seq_bigrams": r2_seq_bigrams
    }
    return metrics
