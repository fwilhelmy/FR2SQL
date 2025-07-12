import os
from typing import List

from .SchemaQuestionLinker import (
    fit_tfidf,
    extract_candidates_yake,
    rank_candidates_by_tfidf,
    fuzzy_match_schema,
)


class DialogModule:
    """Interactive helper to link user questions to the database schema."""

    def __init__(self, schema_elements: List[str], memory_path: str):
        self.schema_elements = schema_elements
        self.memory_path = memory_path
        self.corpus = self._load_memory()

    def _load_memory(self) -> List[str]:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def _save_memory(self) -> None:
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            for q in self.corpus:
                f.write(q + "\n")

    def add_to_memory(self, question: str) -> None:
        self.corpus.append(question)
        self._save_memory()

    def schema_link(self, question: str):
        tfidf = fit_tfidf(self.corpus)
        candidates = extract_candidates_yake(question)
        top_phrases = rank_candidates_by_tfidf(candidates, tfidf)
        links = fuzzy_match_schema(top_phrases, self.schema_elements)
        return links

    def ask(self, prompt: str, attempt: int = 1) -> str:
        """Prompt the user and return the entered text."""
        if attempt > 1:
            print("Please reformulate your request.")
        return input(prompt).strip()

    def ask_question(self, attempt: int = 1) -> str:
        return self.ask("Question: ", attempt)

    def ask_clarification(self, attempt: int = 1) -> str:
        return self.ask("Clarification: ", attempt)

    def run(self) -> None:
        attempt = 1
        while True:
            question = self.ask_question(attempt)
            if question.lower() in {"exit", "quit"}:
                break
            links = self.schema_link(question)
            for m in links:
                print(f"{m['keyword']} -> {m['schema_element']} ({m['score']}%)")
            self.add_to_memory(question)
            attempt += 1


if __name__ == "__main__":
    example_schema = ["employees", "departments"]
    module = DialogModule(example_schema, "memory.txt")
    module.run()
