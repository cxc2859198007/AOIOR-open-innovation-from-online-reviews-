import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from setup import configure_settings, select_free_gpus
from classifier import Classifier
from dataset import TextDataset
from evaluation import evaluate


def prediction(args):
    # Step 1: setup configurations
    configure_settings(args)

    # Step 2: tokenizer and classifier
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    classifier = Classifier(args.model, args.label_num)

    # Step 3: GPU
    free_gpus = select_free_gpus(args.gpu_num)
    device = torch.device(f"cuda:{free_gpus[0]}")
    if args.checkpoint == "":
        raise ValueError("Checkpoint path is required for prediction.")
    checkpoint = torch.load(
        os.path.join("./fine_tuned_model", args.checkpoint, f"{args.base_model}_cp.pth"), map_location=device
    )
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.to(device)
    if args.gpu_num > 1:
        classifier = nn.DataParallel(classifier, device_ids=free_gpus)

    # Step 4: load the test dataset
    test_set = TextDataset(tokenizer, args, 'test')
    test_dataloader = DataLoader(test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=args.cpu_num,
                                 pin_memory=True,
                                 collate_fn=test_set.collate_fn
                                 )

    # Step 5: evaluate predictions
    classifier.eval()
    metrics = evaluate(device, test_dataloader, classifier, args.max_steps + 100)
    results_path = os.path.join("./log", args.checkpoint, "test_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({"test_results": metrics}, f, indent=4)

    print("Prediction Evaluated!")
