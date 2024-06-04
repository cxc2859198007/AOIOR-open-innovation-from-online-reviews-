import os
import json
import math
import random
import numpy as np
import torch
import GPUtil


def check_gpu():
    assert torch.cuda.is_available()


def select_free_gpus(num_required):
    available_gpus = GPUtil.getAvailable(order='memory', limit=num_required, maxLoad=0.5, maxMemory=0.5)
    if len(available_gpus) < num_required:
        raise ValueError(f"Not enough GPUs. Required: {num_required}, Available: {len(available_gpus)}")

    return available_gpus


def to_cuda(batch, device):
    for n in batch.keys():
        if n in ["input_ids", "attention_mask", "label"]:
            batch[n] = batch[n].to(device)


def max_steps(args):
    with open(args.train_dataset, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_size = len(data)
    step_num = math.ceil((data_size * args.epoch) / (args.batch_size * args.grad_accumulate))

    print(f"Data Size = {data_size}")
    print(f"Step Number = {step_num}")

    return step_num


def focal_loss_alpha(args):
    with open(args.train_dataset, 'r', encoding='utf-8') as file:
        data = json.load(file)

    labels = ["characters", "items", "environment", "narrative", "audio", "technical", "improvements"]
    label_counts = {label: 0 for label in labels}

    for review in data.values():
        for label in labels:
            if review[label] == 1:
                label_counts[label] += 1

    data_size = len(data)
    alpha_values = [data_size / label_counts[label] if label_counts[label] > 0 else 0 for label in labels]
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float)

    return alpha_tensor


def configure_settings(args):
    args.label_num = getattr(args, "label_num", 7)
    args.train_dataset = getattr(args, "train_dataset", "aspect_classification/datasets/train_6.json")
    args.dev_dataset = getattr(args, "dev_dataset", "aspect_classification/datasets/dev_set.json")
    args.test_dataset = getattr(args, "test_dataset", "aspect_classification/datasets/test_set.json")

    args.base_model = getattr(args, "base_model", "roberta-large")
    args.model = getattr(args, "model", "xingchenc/roberta-large-finetuned-steam-reviews")
    args.max_length = getattr(args, "max_length", 160)
    args.epoch = getattr(args, 'epoch', 10)
    args.batch_size = getattr(args, 'batch_size', 30)
    args.grad_accumulate = getattr(args, "grad_accumulate", 2)

    args.max_lr = getattr(args, "max_lr", 2e-5)
    args.init_lr = getattr(args, "init_lr", args.max_lr / 25)
    args.weight_decay = getattr(args, "weight_decay", 0.01)
    args.grad_norm = getattr(args, "grad_norm", 1.0)

    args.is_focal = getattr(args, "is_focal", True)
    args.gamma = getattr(args, "gamma", 2.0)
    args.alpha = getattr(args, "alpha", focal_loss_alpha(args))

    args.max_steps = getattr(args, 'max_steps', max_steps(args))
    args.evaluate_interval = getattr(args, "evaluate_interval", math.ceil(args.max_steps / 40))

    args.cpu_num = os.cpu_count() - 2
    args.gpu_num = 1

    args.seed = getattr(args, "seed", 42)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
