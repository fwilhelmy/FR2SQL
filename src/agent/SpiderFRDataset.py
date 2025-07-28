import json
import os
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class SpiderFRDataset(Dataset):
    """Dataset for the Spider-FR JSON files used in training."""

    SPLIT_FILES = {
        "train": "train_spider.json",
        "train_others": "train_others.json",
        "dev": "dev_spider.json",
    }

    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        question_max_length: int = 128,
        query_max_length: int = 256,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if split not in self.SPLIT_FILES:
            raise ValueError(f"Unsupported split '{split}'")

        self.tokenizer = tokenizer
        self.question_max_length = question_max_length
        self.query_max_length = query_max_length

        file_path = os.path.join(data_dir, self.SPLIT_FILES[split])
        with open(file_path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        questions = [rec.get("question", "") for rec in raw_records]
        queries = [rec.get("query", "") for rec in raw_records]

        enc_questions = tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=question_max_length,
            return_tensors="pt",
        )

        enc_queries = tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=query_max_length,
            return_tensors="pt",
        )

        self.input_ids = enc_questions.input_ids
        self.attention_mask = enc_questions.attention_mask
        self.labels = enc_queries.input_ids

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
