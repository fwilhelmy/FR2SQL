from __future__ import annotations

import os

from tqdm import tqdm

from agent.SpiderFRDataset import SpiderFRDataset


def populate_dialog_memory(
    dataset: SpiderFRDataset,
    db_root: str = "databases/spider/test_database",
) -> None:
    """Append each question to the dialog memory of its database."""

    memories = {}
    for question, db_id in tqdm(
        zip(dataset.questions, dataset.db_ids), total=len(dataset.questions), desc="Populating dialog memory"
    ):
        if db_id not in memories:
            memories[db_id] = []
        memories[db_id].append(question.strip())

    for db_id, questions in tqdm(memories.items(), total=len(memories), desc="Writing dialog memory"):
        memory_path = os.path.join(db_root, db_id, "dialog_memory.txt")
        if not os.path.exists(os.path.dirname(memory_path)):
            print(f"Found no directory for database {db_id}")
            continue
        with open(memory_path, "a", encoding="utf-8") as f:
            f.writelines(q + "\n" for q in questions)

if __name__ == "__main__":
    ds = SpiderFRDataset()
    populate_dialog_memory(ds)


