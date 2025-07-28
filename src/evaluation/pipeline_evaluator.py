import json
import os
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from DialogModule import DialogModule
from agent.PromptGenerator import generate_sql_prompt
from spider.process_sql import tokenize

MODEL_DIR = "./adapters"
DB_ROOT = "databases/spider/test_database"


def normalize_sql(sql: str) -> str:
    """Normalize SQL string by tokenizing and joining tokens."""
    tokens = tokenize(sql)
    return " ".join(tokens)


def load_schema(db_id: str, db_root: str) -> Dict[str, List[str]]:
    """Load table/column schema from a Spider JSON file."""
    path = os.path.join(db_root, db_id, f"{db_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tables = meta.get("table_names_original") or meta.get("table_names")
    columns = meta.get("column_names_original") or meta.get("column_names")
    schema: Dict[str, List[str]] = {t: [] for t in tables}

    for tbl_idx, col_name in columns:
        if tbl_idx >= 0:
            schema[tables[tbl_idx]].append(col_name)
    return schema


def build_dialog(schema: Dict[str, List[str]]) -> DialogModule:
    schema_elements: List[str] = list(schema.keys()) + [
        f"{col} {table}" for table, cols in schema.items() for col in cols
    ]
    return DialogModule(schema_elements)


def generate_sql(question: str, schema: Dict[str, List[str]], dialog: DialogModule,
                 model, tokenizer) -> str:
    matches = dialog.schema_link(question)
    matches.sort(key=lambda m: m["score"], reverse=True)

    selected_tables = []
    for m in matches:
        meta = m["schema_element"].split(" ")
        if len(meta) > 1:
            table = meta[1]
        else:
            table = meta[0]
        if table not in selected_tables:
            selected_tables.append(table)

    schema_for_prompt = {"tables": []}
    for t in selected_tables:
        if t in schema:
            schema_for_prompt["tables"].append({"name": t, "columns": schema[t]})

    prompt = generate_sql_prompt(schema_for_prompt, question, db_type="sqlite")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def evaluate_dataset(dataset_path: str,
                      model_dir: str = MODEL_DIR,
                      db_root: str = DB_ROOT,
                      pred_file: str = "pred.txt",
                      label_file: str = "labels.txt",
                      result_file: str = "eval_result.txt") -> float:
    """Evaluate the model on a Spider‑FR style dataset.

    This function now also writes three artefacts:
      * ``label_file`` – SQL labels for each datapoint.
      * ``pred_file`` – predicted SQL queries in the same order.
      * ``result_file`` – summary of the evaluation (currently the exact match
        accuracy).
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

    schema_cache: Dict[str, Dict[str, List[str]]] = {}
    dialog_cache: Dict[str, DialogModule] = {}

    total = len(data)
    correct = 0
    predictions: List[str] = []
    labels: List[str] = []

    for rec in data:
        question = rec["question"]
        expected = rec["query"]
        db_id = rec["db_id"]

        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(db_id, db_root)
            dialog_cache[db_id] = build_dialog(schema_cache[db_id])

        schema = schema_cache[db_id]
        dialog = dialog_cache[db_id]
        predicted = generate_sql(question, schema, dialog, model, tokenizer)

        predictions.append(predicted)
        labels.append(expected)

        if normalize_sql(predicted) == normalize_sql(expected):
            correct += 1

    accuracy = correct / total if total else 0.0
    result_line = f"Exact match accuracy: {accuracy:.2%} ({correct}/{total})"

    # Write artefacts
    with open(pred_file, "w", encoding="utf-8") as pf:
        pf.write("\n".join(predictions))
    with open(label_file, "w", encoding="utf-8") as lf:
        lf.write("\n".join(labels))
    with open(result_file, "w", encoding="utf-8") as rf:
        rf.write(result_line + "\n")

    print(result_line)
    return accuracy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SQL generation model")
    parser.add_argument("dataset", help="Path to Spider-FR JSON dataset")
    parser.add_argument("--model", default=MODEL_DIR, help="Model directory")
    parser.add_argument("--db-root", default=DB_ROOT, help="Root directory of test databases")
    parser.add_argument("--pred-file", default="pred.txt", help="File to write predictions")
    parser.add_argument("--label-file", default="labels.txt", help="File to write gold labels")
    parser.add_argument("--result-file", default="eval_result.txt", help="File to write evaluation result")
    args = parser.parse_args()

    evaluate_dataset(
        args.dataset,
        args.model,
        args.db_root,
        pred_file=args.pred_file,
        label_file=args.label_file,
        result_file=args.result_file,
    )
