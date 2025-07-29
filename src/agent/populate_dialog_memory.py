from __future__ import annotations

import os
from typing import Iterable

from tqdm import tqdm

from agent.SpiderFRDataset import SpiderFRDataset


def populate_dialog_memory(
    dataset: SpiderFRDataset,
    db_root: str = "databases/spider/test_database",
) -> None:
    """Append each question to the dialog memory of its database."""

    for question, db_id in tqdm(
        zip(dataset.questions, dataset.db_ids), total=len(dataset.questions)
    ):
        memory_path = os.path.join(db_root, db_id, "dialog_memory.txt")
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, "a", encoding="utf-8") as f:
            f.write(question.strip() + "\n")


if __name__ == "__main__":
    ds = SpiderFRDataset()
    populate_dialog_memory(ds)


