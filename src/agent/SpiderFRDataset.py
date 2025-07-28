import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SpiderFRDataset(Dataset):
    """Hardcoded loader for the Spider-FR training split."""

    def __init__(self):
        # Read the training split
        with open("./data/spider-fr/train_spider.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [rec["question"] for rec in data]
        queries = [rec["query"] for rec in data]

        tokenizer = AutoTokenizer.from_pretrained(
            "google/mt5-small",
            use_fast=False,
        )

        enc_questions = tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc_queries = tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).input_ids

        self.input_ids = enc_questions.input_ids
        self.attention_mask = enc_questions.attention_mask
        self.labels = enc_queries

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
