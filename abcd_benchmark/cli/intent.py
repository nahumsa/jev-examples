"""CLI for reproducible ABCD intent-classification experiments."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from abcd_benchmark.intent.evaluation import (
    EvaluationConfig,
    run_evaluation,
    select_conversations,
)
from abcd_benchmark.intent.summary import summarize

# Preserve imports used by older callers and tests.
__all__ = ['main', 'select_conversations', 'summarize']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['sample', 'train', 'dev', 'test'], default='dev')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--limit', type=int, default=20, help='Maximum conversations/API runs')
    scope.add_argument('--full-split', action='store_true', help='Evaluate every conversation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--customer-turns', type=int, default=3,
                        help='Stop after this many customer messages; 0 uses all speech')
    parser.add_argument('--model', default='typesafe:jev-latest')
    parser.add_argument('--data-dir', type=Path, default=Path('data/abcd'))
    parser.add_argument('--dataset', type=Path, help='Use an existing ABCD JSON or JSON.gz')
    parser.add_argument('--output', type=Path, default=Path('results/abcd.jsonl'))
    parser.add_argument('--dry-run', action='store_true', help='Prepare requests without inference')
    parser.add_argument('--concurrency', type=int, default=1, help='Maximum parallel API runs')
    args = parser.parse_args()
    if args.limit < 1 or args.customer_turns < 0 or args.concurrency < 1:
        parser.error('--limit and --concurrency must be positive; --customer-turns nonnegative')
    if not args.dry_run and not os.environ.get('TYPESAFE_API_KEY'):
        parser.error('Set TYPESAFE_API_KEY to run Jev, or use --dry-run (no API calls)')
    config = EvaluationConfig(
        **{k: v for k, v in vars(args).items() if k not in {'full_split', 'limit'}},
        limit=None if args.full_split else args.limit,
    )
    print(json.dumps(asyncio.run(run_evaluation(config)), indent=2))


if __name__ == '__main__':
    main()
