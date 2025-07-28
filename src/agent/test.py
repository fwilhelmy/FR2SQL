import json
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# 1) Read & normalize
def read_split(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions, queries = [], []
    for rec in data:
        questions.append(rec["question"])
        queries.append(rec["query"])
    return questions, queries

train_questions, train_queries = read_split("./data/spider-fr/train_spider.json")
dev_questions, dev_queries = read_split("./data/spider-fr/dev_spider.json")

# 2) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False    # <— force the Python/SentencePiece tokenizer
)

# Tokenize the training data
enc_train = tokenizer(
    train_questions,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
lab_train = tokenizer(
    train_queries,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
).input_ids
train_ds = TensorDataset(
    enc_train.input_ids,
    enc_train.attention_mask,
    lab_train
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# Tokenize the dev data
enc_dev = tokenizer(
    dev_questions,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
lab_dev = tokenizer(
    dev_queries,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
).input_ids
dev_ds = TensorDataset(
    enc_dev.input_ids,
    enc_dev.attention_mask,
    lab_dev
)
dev_loader = DataLoader(dev_ds, batch_size=16, shuffle=True)

print("TEST DATASET:")