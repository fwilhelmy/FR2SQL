"""Interactive demo pipeline linking natural language questions to a SQL
schema using ``DialogModule`` and producing a prompt for the future LLM agent."""

import os
from typing import List, Dict
import sqlite3

import pyperclip

from DBManager import DBManager
from DialogModule import DialogModule
from agent.PromptGenerator import generate_sql_prompt
from agent import SimpleAgent

# Base path where Spider test databases are stored
DB_BASE_PATH = "databases/spider/test_database"

# Threshold used to decide whether we are confident enough in the
# schema linking step. This is separate from the TRESHOLD constant of
# the linker which filters individual matches.
CONFIDENCE_THRESHOLD = 80

def compute_average(scores: List[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0

def main() -> None:
    """Run the end‑to‑end demo pipeline."""

    mode = input("Enter the agent ID [Manual, (FlanT5)]: ").strip().lower() or "flant5"
    if mode == "manual": agent = None
    elif mode == "flant5": agent = SimpleAgent()
    else:
        print(f"Unknown command: {mode}")
        print("Available options: Manual, FlanT5")
        return


    db_id = input("Enter the database ID [aircraft]: ").strip() or 'aircraft'
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        print(f"Database '{db_path}' not found.")
        return
    db = DBManager(db_path)
    schema_pairs: Dict[str, List[str]] = db.extract_column_table_pairs()

    # Flatten table and column names for the linker
    schema_elements: List[str] = list(schema_pairs.keys()) + [
        f"{col} {table}" for table, cols in schema_pairs.items() for col in cols
    ]

    # File storing past user questions for the DialogModule
    mem = os.path.join(db_path.rsplit("\\",1)[0], "dialog_memory.txt")
    dialog = DialogModule(schema_elements, mem, mode='normal')

    while True:
        # Ask the user for a new question
        question = dialog.ask(prefix="Question (type 'exit' to quit): ")
        if question.lower() in {"exit", "quit"}:
            break

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
                print(f"Matched: '{m['keyword']}' → '{m['schema_table']}.{m.get('schema_column', '')}' ({m['score']}%)")

        avg_score = compute_average([m["score"] for m in selected])

        if avg_score < CONFIDENCE_THRESHOLD:
            # Confidence too low → ask for clarification
            question += " " + dialog.ask(prompt="Could you clarify your question?", prefix="Clarification: ")

        # Determine which tables were referenced
        table_metadata = {t: db.extract_table_metadata(t) for t in selected_tables}

        schema_for_prompt = {"tables": []}
        for t, meta in table_metadata.items():
            schema_for_prompt["tables"].append({
                "name": t,
                "columns": meta["columns"],
            })

        prompt = generate_sql_prompt(schema_for_prompt, question)
        pyperclip.copy(prompt)
        print("Prompt copied to clipboard.")

        if agent:
            try:
                generated_sql = agent.generate(prompt)
                if not generated_sql.strip().endswith(";"): generated_sql += ";"
                print("Agent : " + generated_sql + "\n")
            except Exception as exc:
                print(f"Error generating SQL: {exc}")
                continue
        extra = "Press Enter to run the generated query or paste another SQL:\n" if agent else "Enter your SQL query:\n"
        user_sql = input(extra + "SQL> ").strip()
        if not user_sql: user_sql = generated_sql or ""

        # Basic validation of the SQL statement
        if not sqlite3.complete_statement(user_sql):
            print("Invalid or incomplete SQL statement.")
        else:
            try:
                result = db.execute_query(user_sql)
                if "columns" in result:
                    clean_cols = [col.replace("_", " ") for col in result["columns"]]
                    header = " | ".join(clean_cols)
                    print(header)
                    print("-" * len(header))
                    for row in result["rows"]:
                        print(" | ".join(str(v) for v in row))
                else:
                    print(f"Rows affected: {result['rowcount']}")
            except Exception as exc:
                print(f"Error executing query: {exc}")

    db.close()

if __name__ == "__main__":
    main()
