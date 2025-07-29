from __future__ import annotations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from agent import generate_sql_prompt

class BaseModel:
    """Lightweight NL2SQL agent using an off-the-shelf model.

    The agent loads a multilingual instruction-tuned model and uses it
    directly to generate SQL queries from natural language without any
    additional fine-tuning. Questions may be written in either French or
    English.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, schema: dict, question: str, db_type: str = "SQLite") -> str:
        """Generate a SQL query for *question* given the database *schema*."""
        assert hasattr(self, 'model'), "Model must be loaded before generating SQL queries."
        prompt = generate_sql_prompt(schema, question, db_type)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
