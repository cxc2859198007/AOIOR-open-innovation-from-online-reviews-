import json
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, process_type):
        self.tokenizer = tokenizer
        self.max_length = args.max_length

        if process_type == 'train':
            data_file = args.train_dataset
        elif process_type == 'dev':
            data_file = args.dev_dataset
        elif process_type == 'test':
            data_file = args.test_dataset
        else:
            raise ValueError("Invalid Process Type.")

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if process_type == 'train':
            keys_to_keep = list(data.keys())[:-500]
            self.data = {key: data[key] for key in keys_to_keep}
        else:
            self.data = data

        self.ids = list(self.data.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = self.data[self.ids[index]]
        text = item["text"].lower()
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label = torch.tensor([int(ch) for ch in item["label"]], dtype=torch.float)

        return {
            "input_ids": encoded_text['input_ids'].squeeze(0),
            "attention_mask": encoded_text['attention_mask'].squeeze(0),
            "label": label
        }

    def collate_fn(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        label = torch.stack([item["label"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }
