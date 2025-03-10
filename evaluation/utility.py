import os
import random
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.001
EPOCHS = 25
LABEL_IDX_LIST = list(range(25))
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(SEED)
# LSTM model definition


class DiagnosisModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(DiagnosisModel, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, hidden_dim, num_layers=2,
                                  dropout=0.5, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True)
        forward_output = output[range(
            len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        backward_output = output[:, 0, LSTM_HIDDEN_DIM:]
        combined_output = torch.cat((forward_output, backward_output), dim=1)
        return torch.sigmoid(self.fc(combined_output))

# Utility evaluation for MIMIC-III


def evaluate_synthetic_training_mimiciii(real_data, synthetic_data):
    results = {}
    accuracies, f1_scores = [], []
    for label_index in LABEL_IDX_LIST:
        model = DiagnosisModel(
            input_dim=6984, hidden_dim=LSTM_HIDDEN_DIM).to(DEVICE)
        acc, f1 = train_and_evaluate(
            model, synthetic_data, real_data, label_index)
        accuracies.append(acc)
        f1_scores.append(f1)
    results = {
        "accuracy": {"mean": np.mean(accuracies), "std": np.std(accuracies)},
        "f1_score": {"mean": np.mean(f1_scores), "std": np.std(f1_scores)},
    }
    return results

# Helper function to train and evaluate the model


def train_and_evaluate(model, train_data, test_data, label_index):
    train_labels = np.array([p['labels'][label_index] for p in train_data])
    test_labels = np.array([p['labels'][label_index] for p in test_data])
    train_visits = np.array([p['visits'] for p in train_data])
    test_visits = np.array([p['visits'] for p in test_data])

    train_data_tensor = torch.tensor(
        train_visits, dtype=torch.float32).to(DEVICE)
    train_labels_tensor = torch.tensor(
        train_labels, dtype=torch.float32).to(DEVICE)
    test_data_tensor = torch.tensor(
        test_visits, dtype=torch.float32).to(DEVICE)
    test_labels_tensor = torch.tensor(
        test_labels, dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(train_data_tensor, torch.tensor(
            [len(v) for v in train_visits]))
        loss = loss_fn(output, train_labels_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(test_data_tensor, torch.tensor(
            [len(v) for v in test_visits]))
        predictions = predictions.cpu().numpy().round()
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
    return accuracy, f1


def utility_evaluation(real_data, synthetic_data):
    return evaluate_synthetic_training_mimiciii(real_data, synthetic_data)
