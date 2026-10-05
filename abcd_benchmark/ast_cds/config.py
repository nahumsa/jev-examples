"""Configuration and validation for AST/CDS runs."""

from dataclasses import dataclass
from pathlib import Path

from .schema import Task


@dataclass(frozen=True)
class BenchmarkConfig:
    task: Task
    output: Path
    split: str = 'dev'
    limit: int | None = 3
    seed: int = 42
    model: str = 'jev-1.13.0'
    data_dir: Path = Path('data/abcd')
    dataset: Path | None = None
    dry_run: bool = False
    concurrency: int = 4
    example_limit: int | None = None
    preprocessing: str = 'reference'

    def validate(self) -> None:
        if self.task not in {'ast', 'cds'}:
            raise ValueError('task must be ast or cds')
        if self.concurrency < 1 or any(v is not None and v < 1 for v in [self.limit, self.example_limit]):
            raise ValueError('limits and concurrency must be positive')
        if self.preprocessing not in {'reference', 'raw'}:
            raise ValueError('preprocessing must be reference or raw')
