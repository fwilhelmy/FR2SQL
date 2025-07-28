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

## Usage

### Training

Fine‑tune the multilingual model on the French Spider dataset:

```bash
python src/agent/train.py
```

Adapters and tokenizer files will be saved in the `adapters/` directory.

### Interactive Demo

Run the end‑to‑end pipeline that links a natural language question to a SQLite database and executes the generated query:

```bash
python src/main.py
```

This uses `database/sqlite/employee_db.sqlite` and stores past queries in `data/dialog_memory.txt`.

### Evaluation

Evaluate a trained model on a Spider‑FR style dataset. In addition to exact
match, the script will attempt to run the official
[test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) evaluation
if the repository is present under ``test-suite-sql-eval/``:

```bash
python src/evaluation/pipeline_evaluator.py data/spider-fr/dev_spider.json --model adapters --db-root databases/spider/test_database
```
The script writes the predictions and an accuracy report in the current directory
and, when the test-suite repo is available, prints the Test Suite execution
accuracy.

## Future Work

Implementation of the training and inference pipeline is planned but not yet committed. The project schedule includes dataset preparation, fine‑tuning, evaluation, and reporting, as detailed in the project proposal.

## Credits

This project makes use of the following open source resources:

- **Spider** dataset and evaluation tools from [taoyds/spider](https://github.com/taoyds/spider).  
  Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang and Dragomir Radev. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*, 2019. <https://arxiv.org/abs/1809.08887>
- **Test Suite Evaluation** code from [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval).  
  Ruiqi Zhong, Tao Yu and Dan Klein. *Semantic Evaluation for Text-to-SQL with Distilled Test Suites*, 2020. <https://arxiv.org/abs/2010.02840>
- **Spider‑FR** dataset from [Marchanjo/spider-fr](https://huggingface.co/datasets/Marchanjo/spider-fr).  
  Marcelo Archanjo Jose and Fabio Gagliardi Cozman. *A multilingual translator to SQL with database schema pruning to improve self-attention*, 2023. <https://doi.org/10.1007/s41870-023-01342-3>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
