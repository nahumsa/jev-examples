"""Pure assembly of intent run metadata and aggregate metrics."""

from dataclasses import asdict
from pathlib import Path

from abcd_benchmark.common.data import DATA_REVISION

from .config import EvaluationConfig
from .datasets import PreparedEvaluation
from .metrics import summarize
from .schema import Intent


def build_summary(config: EvaluationConfig, prepared: PreparedEvaluation, rows: list[dict],
                  *, complete: bool, wall_time_seconds: float) -> dict:
    return {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        'data_revision': DATA_REVISION, 'dataset': str(prepared.dataset),
        'dry_run': config.dry_run, 'output': str(config.output),
        'complete': complete, 'expected_conversations': len(prepared.rows),
        'wall_time_seconds': wall_time_seconds,
        **({'prepared_requests': len(rows), 'intent_options': len(Intent)}
           if config.dry_run else summarize(rows, prepared.intent_to_flow)),
    }
