import os
import torch
import pickle
import json
import random
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
# pdb.set_trace()

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
LR = 0.001
EPOCHS = 50
LABEL_IDX_LIST = list(range(25))
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500
# CODE_VOCAB_SIZE = 6984
N_CTX = 48


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class DiagnosisModel(nn.Module):
    def __init__(self, code_vocab_size):
        super(DiagnosisModel, self).__init__()
        self.embedding = nn.Linear(code_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=LSTM_HIDDEN_DIM,
                            num_layers=2,
                            dropout=0.5,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*LSTM_HIDDEN_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(
            visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        out_reverse = output[:, 0, LSTM_HIDDEN_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.fc(out_combined)
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)

        return prob


def get_batch(code_vocab_size, ehr_dataset, loc, batch_size, label_idx):
    ehr = ehr_dataset[loc:loc+batch_size]
    batch_ehr = np.zeros((len(ehr), N_CTX, code_vocab_size))
    batch_labels = np.array([p['labels'][label_idx] for p in ehr])
    batch_lens = np.zeros(len(ehr))
    for i, p in enumerate(ehr):
        visits = p['visits'][:N_CTX]
        batch_lens[i] = min(len(visits), N_CTX)
        for j, v in enumerate(visits):
            batch_ehr[i, j][v] = 1

    return batch_ehr, batch_labels, batch_lens


def train_model(code_vocab_size, model, train_dataset, val_dataset, save_name, label_idx):
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    for e in tqdm(range(EPOCHS)):
        np.random.shuffle(train_dataset)
        train_losses = []
        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()
            batch_ehr, batch_labels, batch_lens = get_batch(code_vocab_size,
                train_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(
                batch_labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        cur_train_loss = np.mean(train_losses)
        # print("Epoch %d Training Loss:%.5f" % (e, cur_train_loss))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch(code_vocab_size,
                    val_dataset, v_i, BATCH_SIZE, label_idx)
                batch_ehr = torch.tensor(
                    batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(
                    batch_labels, dtype=torch.float32).to(device)
                prob = model(batch_ehr, batch_lens)
                val_loss = bce(prob, batch_labels)
                val_losses.append(val_loss.cpu().detach().numpy())
            cur_val_loss = np.mean(val_losses)
            # print("Epoch %d Validation Loss:%.5f" % (e, cur_val_loss))
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, f'{save_name}')
                # print('------------ Save best model ------------')

    model.load_state_dict(state['model'])


def test_model(code_vocab_size, model, test_dataset, label_idx):
    loss_list = []
    pred_list = []
    true_list = []
    bce = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):
            batch_ehr, batch_labels, batch_lens = get_batch(code_vocab_size,
                test_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(
                batch_labels, dtype=torch.float32).to(device)
            prob = model(batch_ehr, batch_lens)
            val_loss = bce(prob, batch_labels)
            loss_list.append(val_loss.cpu().detach().numpy())
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())

    round_list = np.around(pred_list)

    # Extract, save, and display test metrics
    avg_loss = np.mean(loss_list)
    cmatrix = metrics.confusion_matrix(true_list, round_list)
    acc = metrics.accuracy_score(true_list, round_list)
    prc = metrics.precision_score(true_list, round_list)
    rec = metrics.recall_score(true_list, round_list)
    f1 = metrics.f1_score(true_list, round_list)
    auroc = metrics.roc_auc_score(true_list, pred_list)
    (precisions, recalls, _) = metrics.precision_recall_curve(true_list, pred_list)
    auprc = metrics.auc(recalls, precisions)

    metrics_dict = {}
    metrics_dict['Test Loss'] = avg_loss
    metrics_dict['Confusion Matrix'] = cmatrix
    metrics_dict['Accuracy'] = acc
    metrics_dict['Precision'] = prc
    metrics_dict['Recall'] = rec
    metrics_dict['F1 Score'] = f1
    metrics_dict['AUROC'] = auroc
    metrics_dict['AUPRC'] = auprc

    return metrics_dict


def utility_from_halo_evaluation(code_vocab_size, args, synthetic_dataset):
    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    with open(f"data/{args.dataset}/{args.dataset_version}/idToLabel.json", 'r') as f:
        id_to_label = json.load(f)

    train_ehr_dataset = pickle.load(
        open(f'data/{args.dataset}/{args.dataset_version}/train.pkl', 'rb'))
    val_ehr_dataset = pickle.load(
        open(f'data/{args.dataset}/{args.dataset_version}/val.pkl', 'rb'))
    test_ehr_dataset = pickle.load(
        open(f'data/{args.dataset}/{args.dataset_version}/test.pkl', 'rb'))

    results = {}

    # Placeholders for results
    f1_scores_syn = []
    accuracies_syn = []
    precision_syn = []
    recall_syn = []
    auroc_syn = []
    auprc_syn = []
    
    f1_scores_real = []
    accuracies_real = []
    precision_real = []
    recall_real = []
    auroc_real = []
    auprc_real = []
    utility_model_dir = f'{args.output_path}/utility_models'
    os.makedirs(utility_model_dir, exist_ok=True)
    for i in LABEL_IDX_LIST:
        # for i in tqdm(range(1)):
        print(f"processing label {str(i)}: {id_to_label[str(i)]}")
        label_results = {}

        # Prepare datasets
        syn_pos_label_dataset = [
            p for p in synthetic_dataset if p['labels'][i] == 1]
        syn_neg_label_dataset = [
            p for p in synthetic_dataset if p['labels'][i] == 0]
        train_pos_label_dataset = [
            p for p in train_ehr_dataset if p['labels'][i] == 1]
        train_neg_label_dataset = [
            p for p in train_ehr_dataset if p['labels'][i] == 0]
        val_pos_label_dataset = [
            p for p in val_ehr_dataset if p['labels'][i] == 1]
        val_neg_label_dataset = [
            p for p in val_ehr_dataset if p['labels'][i] == 0]
        test_pos_label_dataset = [
            p for p in test_ehr_dataset if p['labels'][i] == 1]
        test_neg_label_dataset = [
            p for p in test_ehr_dataset if p['labels'][i] == 0]

        val_dataset = list(np.random.choice(val_pos_label_dataset, int(NUM_VAL_EXAMPLES/2), replace=(False if len(val_pos_label_dataset)
                                                                                                     >= NUM_VAL_EXAMPLES else True))) + list(np.random.choice(val_neg_label_dataset, int(NUM_VAL_EXAMPLES/2), replace=False))
        test_dataset = list(np.random.choice(test_pos_label_dataset, int(NUM_TEST_EXAMPLES/2), replace=(False if len(test_pos_label_dataset)
                            >= NUM_TEST_EXAMPLES else True))) + list(np.random.choice(test_neg_label_dataset, int(NUM_TEST_EXAMPLES/2), replace=False))

        train_dataset_real = list(np.random.choice(train_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(train_pos_label_dataset)
                                                                                                                >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(train_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))
        train_dataset_syn = list(np.random.choice(syn_pos_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=(False if len(syn_pos_label_dataset)
                                                                                                             >= int(NUM_TRAIN_EXAMPLES/2) else True))) + list(np.random.choice(syn_neg_label_dataset, int(NUM_TRAIN_EXAMPLES/2), replace=False))

        train_dataset_real = [
            p for p in train_dataset_real if len(p['visits']) > 0]
        train_dataset_syn = [
            p for p in train_dataset_syn if len(p['visits']) > 0]
        val_dataset = [p for p in val_dataset if len(p['visits']) > 0]
        test_dataset = [p for p in test_dataset if len(p['visits']) > 0]

        # Perform the different experiments
        print("Training on real data...")
        model_real = DiagnosisModel(code_vocab_size).to(device)
        if not os.path.exists(f"{utility_model_dir}/utility_real_{i}.pt"):
            train_model(code_vocab_size, model_real, train_dataset_real,
                        val_dataset, f"{utility_model_dir}/utility_real_{i}.pt", i)
        state = torch.load(f'{utility_model_dir}/utility_real_{i}.pt', map_location=device)
        model_real.load_state_dict(state['model'])
        test_results_real = test_model(code_vocab_size,model_real, test_dataset, i)
        label_results[f'Real'] = test_results_real

        print("Training on synthetic data...")
        model_syn = DiagnosisModel(code_vocab_size).to(device)
        if not os.path.exists(f"{utility_model_dir}/utility_syn_{i}.pt"):
            train_model(code_vocab_size, model_syn, train_dataset_syn,
                        val_dataset, f"{utility_model_dir}/utility_syn_{i}.pt", i)
        state = torch.load(f'{utility_model_dir}/utility_syn_{i}.pt', map_location=device)
        model_syn.load_state_dict(state['model'])
        test_results_syn = test_model(code_vocab_size, model_syn, test_dataset, i)
        label_results[f'Syn'] = test_results_syn

        results[id_to_label[str(i)]] = label_results

        print(label_results)
    # Assuming `results` is your dictionary with metrics for each label
    final_results = {}
    for label, metrics in results.items():
        final_results[label] = {}
        # Metrics for synthetic data
        # Assuming 'Syn' contains synthetic data results
        syn_metrics = metrics['Syn']
        f1_scores_syn.append(syn_metrics['F1 Score'])
        accuracies_syn.append(syn_metrics['Accuracy'])
        precision_syn.append(syn_metrics['Precision'])
        recall_syn.append(syn_metrics['Recall'])
        auroc_syn.append(syn_metrics['AUROC'])
        auprc_syn.append(syn_metrics['AUPRC'])
        final_results[label]['syn'] = syn_metrics
        # Metrics for real data
        # Assuming 'Real' contains real data results
        real_metrics = metrics['Real']
        f1_scores_real.append(real_metrics['F1 Score'])
        accuracies_real.append(real_metrics['Accuracy'])
        precision_real.append(real_metrics['Precision'])
        recall_real.append(real_metrics['Recall'])
        auroc_real.append(real_metrics['AUROC'])
        auprc_real.append(real_metrics['AUPRC'])
        final_results[label]['real'] = real_metrics
    # Calculate mean and standard deviation for synthetic data
    final_results['overall'] = {}
    final_results['overall']['f1_mean_syn'] = np.mean(f1_scores_syn)
    final_results['overall']['f1_std_syn'] = np.std(f1_scores_syn)
    final_results['overall']['acc_mean_syn'] = np.mean(accuracies_syn)
    final_results['overall']['acc_std_syn'] = np.std(accuracies_syn)
    final_results['overall']['precision_mean_syn'] = np.mean(precision_syn)
    final_results['overall']['precision_std_syn'] = np.std(precision_syn)
    final_results['overall']['recall_mean_syn'] = np.mean(recall_syn)
    final_results['overall']['recall_std_syn'] = np.std(recall_syn)
    final_results['overall']['auroc_mean_syn'] = np.mean(auroc_syn)
    final_results['overall']['auroc_std_syn'] = np.std(auroc_syn)
    final_results['overall']['auprc_mean_syn'] = np.mean(auprc_syn)
    final_results['overall']['auprc_std_syn'] = np.std(auprc_syn)

    # Calculate mean and standard deviation for real data
    final_results['overall']['f1_mean_real'] = np.mean(f1_scores_real)
    final_results['overall']['f1_std_real'] = np.std(f1_scores_real)
    final_results['overall']['acc_mean_real'] = np.mean(accuracies_real)
    final_results['overall']['acc_std_real'] = np.std(accuracies_real)
    final_results['overall']['precision_mean_real'] = np.mean(precision_real)
    final_results['overall']['precision_std_real'] = np.std(precision_real)
    final_results['overall']['recall_mean_real'] = np.mean(recall_real)
    final_results['overall']['recall_std_real'] = np.std(recall_real)
    final_results['overall']['auroc_mean_real'] = np.mean(auroc_real)
    final_results['overall']['auroc_std_real'] = np.std(auroc_real)
    final_results['overall']['auprc_mean_real'] = np.mean(auprc_real)
    final_results['overall']['auprc_std_real'] = np.std(auprc_real)
    return final_results
