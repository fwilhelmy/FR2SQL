"""Interactive demo pipeline linking natural language questions to a SQL
schema using ``DialogModule`` and producing a prompt for the future LLM agent."""

import os
from typing import List, Dict

from DBManager import DBManager
from DialogModule import DialogModule
from agent.PromptGenerator import generate_sql_prompt

DB_PATH = "databases/spider/test_database/bakery_1/bakery_1.sqlite"

# Threshold used to decide whether we are confident enough in the
# schema linking step.  This is separate from the TRESHOLD constant of
# the linker which filters individual matches.
CONFIDENCE_THRESHOLD = 75

def compute_average(scores: List[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0

# "Quel est le salaire moyen des employés par département ?"

def main() -> None:
    """Run the end‑to‑end demo pipeline."""
    
    db = DBManager(DB_PATH)
    schema_pairs: Dict[str, List[str]] = db.extract_column_table_pairs()

    # Flatten table and column names for the linker
    schema_elements: List[str] = list(schema_pairs.keys()) + [
        f"{col} {table}" for table, cols in schema_pairs.items() for col in cols
    ]

    # File storing past user questions for the DialogModule
    mem = os.path.join(DB_PATH.rsplit("/",1)[0], "dialog_memory.txt")
    # Initialize the dialog helper which keeps a history of past questions
    dialog = DialogModule(schema_elements, mem)

    # Ask the user for a new question
    question = dialog.ask()

    # Link the user's request to the database schema
    matches = dialog.schema_link(question)
    matches.sort(key=lambda m: m["score"], reverse=True)

    selected = []
    selected_tables = []
    for m in matches:
        meta = m["schema_element"].split(" ")
        if len(meta) > 1:
            m["schema_table"] = meta[1]
            m["schema_column"] = meta[0]
        else:
            m["schema_table"] = meta[0]
        if m["schema_table"] not in selected_tables and m["score"] > CONFIDENCE_THRESHOLD:
            selected.append(m)
            selected_tables.append(m["schema_table"])
        print(f"Matched: '{m['keyword']}' → '{m['schema_table']}.{m.get('schema_column',"")}' ({m['score']}%)")

    avg_score = compute_average([m["score"] for m in selected])

    if avg_score < CONFIDENCE_THRESHOLD:
        # Confidence too low → ask for clarification
        question += " " + dialog.ask(prompt="Could you clarify your question?", prefix="Clarification: ")

    # Save the (possibly clarified) question for future runs
    dialog.add_to_memory(question)

    # Determine which tables were referenced
    table_metadata = {t: db.extract_table_metadata(t) for t in selected_tables}

    # Convert to the format expected by ContextGenerator
    schema_for_prompt = {"tables": []}
    for t, meta in table_metadata.items():
        schema_for_prompt["tables"].append({
            "name": t,
            "columns": meta["columns"],
        })

    prompt = generate_sql_prompt(schema_for_prompt, question, db_type="sqlite")

    # TODO: integrate the actual LLM call here
    print("\n[LLM PROMPT]\n" + prompt + "\n")

    # TODO Test and execute the generated SQL query

if __name__ == "__main__":
    main()

