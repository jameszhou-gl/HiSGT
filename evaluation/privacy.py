import random
from tqdm import tqdm
from sklearn import metrics
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, pairwise_distances

# Set a fixed seed for reproducibility
LABEL_VOCAB_SIZE = 25
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)


def privacy_evaluation(train_data, test_data, synthetic_data):
    results = {}
    results['MIA'] = membership_inference_attack(
        train_data, test_data, synthetic_data)
    results['AIA'] = attribute_inference_attack(
        train_data, test_data, synthetic_data)
    return results


def membership_inference_attack(train_data, test_data, synthetic_data):
    NUM_TEST_EXAMPLES = 500

    train_data = [(p, 1) for p in train_data]
    test_data = [(p, 0) for p in test_data]
    synthetic_data = [p for p in synthetic_data if len(p['visits']) > 0]

    attack_dataset_pos = list(random.sample(train_data, NUM_TEST_EXAMPLES))
    attack_dataset_neg = list(random.sample(test_data, NUM_TEST_EXAMPLES))
    random.shuffle(attack_dataset_pos)
    random.shuffle(attack_dataset_neg)
    test_attack_dataset = attack_dataset_pos + attack_dataset_neg
    random.shuffle(test_attack_dataset)

    def find_hamming(ehr, dataset):
        min_d = 1e10
        visits = ehr['visits']
        labels = ehr['labels']
        for p in dataset:
            d = 0 if len(visits) == len(p['visits']) else 1
            l = p['labels']
            d += ((labels + l) == 1).sum()
            for i in range(len(visits)):
                v = visits[i]
                if i >= len(p['visits']):
                    d += len(v)
                else:
                    v2 = p['visits'][i]
                    d += len(v) + len(v2) - (2 * len(set(v) & set(v2)))

            min_d = d if d < min_d else min_d
        return min_d

    # Perform the Hamming Distance experiment
    ds = [(find_hamming(ehr, synthetic_data), l)
          for (ehr, l) in tqdm(test_attack_dataset)]
    median_dist = np.median([d for (d, l) in ds])
    preds = [1 if d < median_dist else 0 for (d, l) in ds]
    labels = [l for (d, l) in ds]
    results = {
        "Accuracy": metrics.accuracy_score(labels, preds),
        "Precision": metrics.precision_score(labels, preds),
        "Recall": metrics.recall_score(labels, preds),
        "F1": metrics.f1_score(labels, preds)
    }
    return results


def attribute_inference_attack(train_data, test_data, synthetic_data):
    test_data = [{'labels': p['labels'], 'visits': set(
        [c for v in p['visits'] for c in v])} for p in test_data]
    train_data = [{'labels': p['labels'], 'visits': set(
        [c for v in p['visits'] for c in v])} for p in train_data]
    train_data = np.random.choice(
        train_data, len(test_data), replace=False)
    synthetic_data = [{'labels': p['labels'], 'visits': set(
        [c for v in p['visits'] for c in v])} for p in synthetic_data if len(p['visits']) > 0]
    synthetic_data = np.random.choice(
        synthetic_data, len(test_data), replace=False)

    common_codes = set([cd for cd, _ in Counter(
        [c for p in train_data for c in p['visits']]).most_common()[0:100]])

    test_data = [{'labels': set([c for c in p['labels'].nonzero()[0].tolist()] + [c + LABEL_VOCAB_SIZE for c in p['visits']
                                                                                  if c in common_codes]), 'codes': set([c for c in p['visits'] if c not in common_codes])} for p in test_data]
    train_data = [{'labels': set([c for c in p['labels'].nonzero()[0].tolist()] + [c + LABEL_VOCAB_SIZE for c in p['visits']
                                                                                   if c in common_codes]), 'codes': set([c for c in p['visits'] if c not in common_codes])} for p in train_data]
    synthetic_data = [{'labels': set([c for c in p['labels'].nonzero()[0].tolist()] + [c + LABEL_VOCAB_SIZE for c in p['visits']
                                                                                       if c in common_codes]), 'codes': set([c for c in p['visits'] if c not in common_codes])} for p in synthetic_data]

    def calc_dist(lab1, lab2):
        return len(lab1.union(lab2)) - len(lab1.intersection(lab2))

    def find_closest(patient, data, k):
        cond = patient['labels']
        dists = [(calc_dist(cond, ehr['labels']), ehr['codes'])
                 for ehr in data]
        dists.sort(key=lambda x: x[0], reverse=False)
        options = [o[1] for o in dists[:k]]
        return options

    def calc_attribute_risk(train_dataset, reference_dataset, k):
        tp = 0
        fp = 0
        fn = 0
        for p in tqdm(train_dataset):
            closest_k = find_closest(p, reference_dataset, k)
            pred_codes = set([cd for cd, cnt in Counter(
                [c for p in closest_k for c in p]).items() if cnt > k/2])
            true_pos = len(pred_codes.intersection(p['codes']))
            false_pos = len(pred_codes) - true_pos
            false_neg = len(p['codes']) - true_pos
            tp += true_pos
            fp += false_pos
            fn += false_neg

        f1 = tp / (tp + (0.5 * (fp + fn)))
        return f1

    K = 1
    att_risk = calc_attribute_risk(train_data, synthetic_data, K)
    baseline_risk = calc_attribute_risk(train_data, test_data, K)
    results = {
        "Attribute Attack F1 Score": att_risk,
        "Baseline Attack F1 Score": baseline_risk
    }
    return results
