import json
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split

class SpiderFRDataset(Dataset):
    """Dataset loader for the French version of Spider.

    The loader combines the ``train_spider.json``, ``train_others.json`` and
    ``dev_spider.json`` files from ``data/spider-fr``. Questions and SQL queries
    are tokenized using Spider's ``tokenize`` utility and then converted to
    token IDs with a HuggingFace ``AutoTokenizer``.
    """
    
    DATA_FILES = [
        os.path.join("./data/spider-fr", "train_spider.json"),
        os.path.join("./data/spider-fr", "train_others.json"),
        os.path.join("./data/spider-fr", "dev_spider.json"),
    ]

    def __init__(
        self,
        tokenizer_name: str = "google/mt5-small",
        max_question_length: int = 128,
        max_query_length: int = 256,
    ) -> None:
        self.questions, self.questions_toks = [], []
        self.queries, self.queries_toks = [], []
        self.db_ids = []
        for path in self.DATA_FILES:
            with open(path, "r", encoding="utf-8") as f:
                for rec in json.load(f):
                    self.db_ids.append(rec["db_id"])
                    self.questions.append(rec["question"])
                    self.questions_toks.append(rec["question_toks"])
                    self.queries.append(rec["query_toks"])
                    self.queries_toks.append(rec["query_toks"])

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

        enc_questions = tokenizer(
            self.questions_toks,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_question_length,
            return_tensors="pt",
        )

        enc_queries = tokenizer(
            self.queries_toks,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_query_length,
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
            "db_id": self.db_ids[idx],
        }

def split_and_load(
    dataset: SpiderFRDataset,
    split_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split ``dataset`` into train/validation subsets and return dataloaders."""

    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    dataset = SpiderFRDataset()
    print("Dataset size:", len(dataset))

    train_loader, val_loader = split_and_load(dataset, split_ratio=0.8)
    print("Train set size:", len(train_loader.dataset))
    print("Validation set size:", len(val_loader.dataset))
