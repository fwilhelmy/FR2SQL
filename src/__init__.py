"""Convenience imports for the FR2SQL package.

This module exposes the most commonly used classes and utilities so they
can be imported directly from :mod:`src`.
"""

from .DBManager import DBManager
from .DialogModule import DialogModule

from .agent import (
    SimpleAgent,
    generate_sql_prompt,
    SpiderFRDataset,
    split_and_load,
)

from .evaluation import Evaluator, Picard, evaluate_dataset

__all__ = [
    "DBManager",
    "DialogModule",
    "SimpleAgent",
    "generate_sql_prompt",
    "SpiderFRDataset",
    "split_and_load",
    "Evaluator",
    "Picard",
    "evaluate_dataset",
]

