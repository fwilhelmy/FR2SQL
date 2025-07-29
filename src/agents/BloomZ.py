from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch

class BloomZ:
    def __init__(self) -> None:
        model_name: str = "bigscience/bloomz-7b1-mt"
        cache_dir = f"cache/agents/{model_name.rsplit('/',1)[-1]}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map={"": "cpu"},
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )

    def generate(self, prompt: str) -> str:
        """Generate a SQL query for *question* given the database *schema*."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
