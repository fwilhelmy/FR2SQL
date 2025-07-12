import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class SpiderFRDataset(Dataset):
    """
    Dataset handler for Marchanjo/spider-fr dataset, formatted for causal LM QLoRA training.
    Concatenates question and SQL query, masking question tokens in labels for causal loss.
    """
    def __init__(self,
                 hf_dataset_name: str = "Marchanjo/spider-fr",
                 split: str = "train",
                 tokenizer: PreTrainedTokenizerBase = None,
                 max_length: int = 512,
                 prompt_template: str = "{question}\nSQL:",
                 eos_token: str = None):
        assert tokenizer is not None, "A tokenizer must be provided"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        # Use provided EOS or tokenizer default
        self.eos_token = eos_token if eos_token is not None else (tokenizer.eos_token or "")

        # Load Hugging Face dataset
        self.dataset = datasets.load_dataset(hf_dataset_name, split=split)

        # Tokenize and prepare
        self.dataset = self.dataset.map(
            self._tokenize_example,
            remove_columns=self.dataset.column_names,
        )

        # Set PyTorch format
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def _tokenize_example(self, example):
        # Extract fields
        question = example.get("question", "")
        query = example.get("query", "")

        # Build the prompt and full sequence
        prompt_text = self.prompt_template.format(question=question)
        full_text = prompt_text + " " + query + self.eos_token

        # Tokenize full sequence
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Determine prompt token length (tokens not equal to pad)
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        # Count non-pad tokens in prompt to mask labels
        prompt_token_count = sum(
            1 for tid in prompt_encoding["input_ids"] if tid != self.tokenizer.pad_token_id
        )

        # Create labels: mask prompt tokens with -100
        labels = input_ids.copy()
        labels[:prompt_token_count] = [-100] * prompt_token_count

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_dataset(self):
        """
        Returns the internal Hugging Face dataset formatted for PyTorch.
        """
        return self.dataset
