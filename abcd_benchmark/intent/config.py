"""Configuration and validation for intent-routing evaluations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = 'dev'
    limit: int | None = 20
    seed: int = 42
    customer_turns: int = 3
    model: str = 'typesafe:jev-latest'
    data_dir: Path = Path('data/abcd')
    dataset: Path | None = None
    output: Path = Path('results/abcd.jsonl')
    dry_run: bool = False
    concurrency: int = 1

    def validate(self) -> None:
        if self.limit is not None and self.limit < 1:
            raise ValueError('limit must be positive or None (full split)')
        if self.customer_turns < 0 or self.concurrency < 1:
            raise ValueError('customer_turns must be nonnegative and concurrency positive')
