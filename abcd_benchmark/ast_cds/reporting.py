"""Pure benchmark-summary assembly, including partial-run validity rules."""

from dataclasses import asdict
from pathlib import Path

from abcd_benchmark.common.data import DATA_REVISION

from .config import BenchmarkConfig
from .datasets import PreparedBenchmark
from .metrics import benchmark_metrics
from .schema import TOKENIZER_REVISION


def build_summary(config: BenchmarkConfig, prepared: PreparedBenchmark, rows: list[dict], *,
                  complete: bool, failed_examples: list[dict], prepared_request_count: int,
                  wall_time_seconds: float) -> dict:
    raw = config.preprocessing == 'raw'
    cascading_valid = (config.task == 'cds' and complete and not config.dry_run
                       and len(prepared.examples) == prepared.total_selected_examples)
    summary = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        'protocol': prepared.protocol, 'data_revision': DATA_REVISION,
        'tokenizer': None if raw else 'google-bert/bert-base-uncased',
        'tokenizer_revision': None if raw else TOKENIZER_REVISION,
        'source_sha256': prepared.source_hashes,
        'complete': complete, 'dry_run': config.dry_run,
        'selected_conversations': prepared.selected_conversations,
        'expected_examples': len(prepared.examples),
        'total_selected_examples_before_limit': prepared.total_selected_examples,
        'full_split_evaluation': prepared.full_split_evaluation,
        'failed_examples': failed_examples,
        'cascading_score_valid_for_selected_conversations': cascading_valid,
        'wall_time_seconds': wall_time_seconds,
        'labels': {'intent': prepared.labels.intents, 'nextstep': prepared.labels.nextsteps,
                   'action': prepared.labels.actions, 'enumerable_value': prepared.labels.values},
        'reference_quirks': [
            'Missing value (-1) cannot match argmax; no-value actions fail joint/turn correctness.',
            'CDS multi-slot examples do not expose slot position; duplicate turn counts are stable-sorted.',
            *([] if raw else [
                'Value-copy context length depends on the annotated action, as in reference preprocessing.',
                'History keeps the FIRST 510 BERT tokens and excludes speaker labels.',
            ]),
        ],
        'comparison_note': ('Raw-input ablation: full dialogue/candidates and observed entity placeholders; not reference preprocessing. '
                            if raw else 'Reference-style examples and scoring; not a reproduction of trained paper models. ')
                           + 'Zero-shot Jev, same judgment implementation; no oracle intent or KB masks.',
        'cost_note': 'Estimate covers saved successful examples only, excluding SDK retries and successful heads of failed examples.',
        **({'prepared_examples': len(rows), 'prepared_requests': prepared_request_count}
           if config.dry_run else benchmark_metrics(rows, config.task)),
    }
    if config.task == 'cds' and not cascading_valid and not config.dry_run:
        # Partial conversations do not have valid remaining-length denominators.
        summary['metrics']['Cascading_Score'] = None
        summary['reference_rounded_metrics']['Cascading_Score'] = None
    return summary
