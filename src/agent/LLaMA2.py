from __future__ import annotations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from agent.BaseModel import BaseModel

class LLaMA2(BaseModel):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str | None = None) -> None:
        super().__init__(model_name, device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            load_in_4bit=True, 
            quantization_config=BitsAndBytesConfig(
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        )
        self.model.to(self.device)
