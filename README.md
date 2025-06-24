# FR2SQL

## Overview

FR2SQL is a prototype project aiming to generate SQL queries from natural language requests written in French. The system is intended to simplify data access for business users and is built as part of the MTI820 course. The project leverages a large language model (LLaMA 3 Instruct‑8B) quantized to 4‑bit and fine‑tuned with QLoRA on the Spider‑FR dataset.

## Methodology

The pipeline is implemented in Python using **PyTorch Lightning**. We quantize the base model with **BitsAndBytes**, then fine‑tune low‑rank adapters via **QLoRA**. Generation is constrained by **PICARD** to ensure valid SQL syntax. The system will be evaluated with metrics such as Execution Accuracy, Exact Match, Valid SQL Rate and Valid Efficiency Score.

## Data

We rely on **Spider‑FR**, a French translation of the Spider benchmark, which pairs natural language questions with SQL queries against complex relational schemas. Each entry includes the question, target SQL and the database identifier.

## Directory Structure

```
├── docs/       Project documents (PDF)
├── src/        Source code (currently empty)
├── data/       Placeholder for datasets
├── logs/       Training logs
```

The `docs` folder contains two PDF documents:

- `MTI820_Proposition_de_Projet.pdf` – the original project proposal describing objectives and planning.
- `MTI820_Revue_de_Litterature.pdf` – a literature review on using large language models for BI assistance.

## Future Work

Implementation of the training and inference pipeline is planned but not yet committed. The project schedule includes dataset preparation, fine‑tuning, evaluation, and reporting, as detailed in the project proposal.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
