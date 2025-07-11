from transformers import AutoModelForCausalLM, AutoTokenizer
from picard import Picard
from picard.dataset_readers.spider import load_tables
import torch

# Load tokenizer and model from LoRA adapters
model_path = "./adapters"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="auto")

# Load database schema (Spider format)
tables = load_tables("data/spider/tables.json")

# Initialize PICARD
picard = Picard(
    tokenizer=tokenizer,
    mode="eval",
    db_path="data/spider/database/",
    tables=tables,
)

# Example prompt
prompt = "Liste les noms des étudiants ayant obtenu une note supérieure à 15."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate using PICARD constraints
with picard.patch_generation(model):
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=1,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))