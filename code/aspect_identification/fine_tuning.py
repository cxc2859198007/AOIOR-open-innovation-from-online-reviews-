import os
import json
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datetime import datetime
from setup import configure_settings, select_free_gpus, to_cuda
from dataset import TextDataset
from classifier import Classifier
from focal_loss import FocalLoss
from evaluation import evaluate


def tensor_to_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()


def record_args(file_path, args):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, default=tensor_to_json)


def record_step(file_path, step_cnt, data):
    step_data = {step_cnt: data}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}

    existing_data.update(step_data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)


def fine_tuning(args):
    # Step 1: setup configurations
    configure_settings(args)
    date_time = datetime.now().strftime("%m-%d_%H-%M")

    # record fine-tuned model
    model_dir = f"./fine_tuned_model/{date_time}"
    os.makedirs(model_dir, exist_ok=True)

    # record intermediate results
    log_dir = f"./log/{date_time}"
    os.makedirs(log_dir, exist_ok=True)
    setup_path = os.path.join(log_dir, "setup.json")
    loss_path = os.path.join(log_dir, "loss.json")
    metrics_path = os.path.join(log_dir, "metrics.json")

    record_args(setup_path, args)

    # Step 2: tokenizer and classifier
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    classifier = Classifier(args.model, args.label_num)

    # Step 3: GPU
    free_gpus = select_free_gpus(args.gpu_num)
    device = torch.device(f"cuda:{free_gpus[0]}")
    if args.checkpoint != "":
        checkpoint = torch.load(
            os.path.join("./fine_tuned_model", args.checkpoint, f"{args.base_model}_cp.pth"), map_location=device
        )
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.to(device)
    if args.gpu_num > 1:
        classifier = nn.DataParallel(classifier, device_ids=free_gpus)

    # Step 4: train mode
    classifier.train()

    # Step 5: load the Train & Dev dataset
    train_set = TextDataset(tokenizer, args, 'train')
    train_dataloader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.cpu_num,
                                  pin_memory=True,
                                  collate_fn=train_set.collate_fn
                                  )

    dev_set = TextDataset(tokenizer, args, 'dev')
    dev_dataloader = DataLoader(dev_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                drop_last=True,
                                num_workers=args.cpu_num,
                                pin_memory=True,
                                collate_fn=dev_set.collate_fn
                                )

    # Step 6: optimizer
    optimizer = optim.AdamW(classifier.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # Step 7: scheduler
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=args.max_steps)

    # Step 8: loss function
    focal_loss_fn = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    # Step 9: fine-tuning
    accumulate_cnt = 0
    step_cnt = 0
    average_loss = 0

    init_metrics = evaluate(device, dev_dataloader, classifier, step_cnt)
    record_step(metrics_path, step_cnt, init_metrics)
    max_f1 = init_metrics['macro_f1']

    for epoch in range(args.epoch):
        for (i, batch) in enumerate(train_dataloader):
            accumulate_cnt += 1
            print(f"Epoch {epoch} ---- Batch {i} ---- Accumulate {accumulate_cnt} ---- Step {step_cnt}")

            # put data into cuda
            to_cuda(batch, device)

            # compute loss
            logits = classifier(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = focal_loss_fn(logits, batch["label"], args.is_focal)
            loss = loss / args.grad_accumulate
            del logits

            # backward
            loss.backward()
            average_loss += loss.item()
            del loss

            # update trainable parameters
            if accumulate_cnt == args.grad_accumulate:
                # clear accumulate_cnt
                accumulate_cnt = 0

                # normalize gradian
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_norm)

                # update parameters
                optimizer.step()
                optimizer.zero_grad()
                step_cnt += 1
                print("\n  Parameters Updated!")

                # learning rate update
                scheduler.step()

                # report lr, loss
                lr = optimizer.param_groups[0]['lr']
                wandb.log({"Learning Rate": lr}, step=step_cnt)
                wandb.log({"Loss": average_loss}, step=step_cnt)
                print(f"\n  LR = {lr:.6e}, Loss = {average_loss:.6f}")
                record_step(loss_path, step_cnt, average_loss)
                average_loss = 0

                # evaluation
                if step_cnt % args.evaluate_interval == 0:
                    metrics = evaluate(device, dev_dataloader, classifier, step_cnt)
                    record_step(metrics_path, step_cnt, metrics)
                    # record the best model
                    if metrics['macro_f1'] > max_f1:
                        max_f1 = metrics['macro_f1']
                        if isinstance(classifier, nn.DataParallel):
                            state_dict = classifier.module.state_dict()
                        else:
                            state_dict = classifier.state_dict()
                        torch.save({
                            'classifier_state_dict': state_dict,
                            'max_f1_score': max_f1
                        }, os.path.join(model_dir, f"{args.base_model}_cp.pth"))

    print("Fine-tuning Finished!")
