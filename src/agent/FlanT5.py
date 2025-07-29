from __future__ import annotations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from agent.BaseModel import BaseModel

class FlanT5(BaseModel):
    def __init__(self, model_name: str = "google/flan-t5-large", device: str | None = None) -> None:
        super().__init__(model_name, device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=self.device,
        )
        self.model.to(self.device)