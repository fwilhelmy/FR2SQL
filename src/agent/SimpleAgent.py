from __future__ import annotations
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from agent import generate_sql_prompt

class SimpleAgent:
    """Lightweight NL2SQL agent using an off-the-shelf model.

    The agent loads a multilingual instruction-tuned model and uses it
    directly to generate SQL queries from natural language without any
    additional fine-tuning. Questions may be written in either French or
    English.
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str | None = None) -> None:
        model_dir = model_name.rsplit("/", 1)[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"cache/agents/{model_dir}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=f"cache/agents/{model_dir}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str) -> str:
        """Generate a SQL query for *question* given the database *schema*."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()