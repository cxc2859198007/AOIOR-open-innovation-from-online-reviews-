import torch
from setup import to_cuda
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
import numpy as np


def manual_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def evaluate(device, dev_dataloader, classifier, step_cnt):
    all_labels = []
    all_predicts = []

    classifier.eval()
    with torch.no_grad():
        for batch in dev_dataloader:
            to_cuda(batch, device)

            logits = classifier(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predict_labels = (torch.sigmoid(logits) > 0.5).int()

            all_labels.append(batch["label"].cpu())
            all_predicts.append(predict_labels.cpu())

    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_predicts = torch.cat(all_predicts, dim=0).numpy()

    # Accuracy
    accuracy_label = np.mean(all_predicts.flatten() == all_labels.flatten())
    accuracy_sample = np.mean(np.all(all_predicts == all_labels, axis=1))
    # Precision
    micro_precision = precision_score(all_labels, all_predicts, average='micro', zero_division=1.0)
    macro_precision = precision_score(all_labels, all_predicts, average='macro', zero_division=1.0)
    samples_precision = precision_score(all_labels, all_predicts, average='samples', zero_division=1.0)
    # Recall
    micro_recall = recall_score(all_labels, all_predicts, average='micro', zero_division=1.0)
    macro_recall = recall_score(all_labels, all_predicts, average='macro', zero_division=1.0)
    samples_recall = recall_score(all_labels, all_predicts, average='samples', zero_division=1.0)
    # F1 score
    micro_f1 = f1_score(all_labels, all_predicts, average='micro', zero_division=1.0)
    macro_f1 = f1_score(all_labels, all_predicts, average='macro', zero_division=1.0)
    samples_f1 = f1_score(all_labels, all_predicts, average='samples', zero_division=1.0)

    # Report
    wandb.log({"Accuracy_Label": accuracy_label}, step=step_cnt)
    wandb.log({"Accuracy_Sample": accuracy_sample}, step=step_cnt)
    wandb.log({"Micro_Precision": micro_precision}, step=step_cnt)
    wandb.log({"Micro_Recall": micro_recall}, step=step_cnt)
    wandb.log({"Micro_F1": micro_f1}, step=step_cnt)
    wandb.log({"Macro_Precision": macro_precision}, step=step_cnt)
    wandb.log({"Macro_Recall": macro_recall}, step=step_cnt)
    wandb.log({"Macro_F1": macro_f1}, step=step_cnt)
    wandb.log({"Samples_Precision": samples_precision}, step=step_cnt)
    wandb.log({"Samples_Recall": samples_recall}, step=step_cnt)
    wandb.log({"Samples_F1": samples_f1}, step=step_cnt)

    metrics = {
        'accuracy_label': accuracy_label,
        'accuracy_sample': accuracy_sample,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'samples_precision': samples_precision,
        'samples_recall': samples_recall,
        'samples_f1': samples_f1
    }

    classifier.train()

    return metrics
