"""Evaluate Jev on ABCD Action State Tracking or Cascading Dialogue Success."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from abcd_benchmark.ast_cds.runner import BenchmarkConfig, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['ast', 'cds'], required=True)
    parser.add_argument('--split', choices=['sample', 'train', 'dev', 'test'], default='dev')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--limit', type=int, default=3, help='Maximum conversations, not examples')
    scope.add_argument('--full-split', action='store_true')
    parser.add_argument('--example-limit', type=int, help='Smoke-test cap; disables CDS cascading score')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='jev-1.13.0', help='TypeSafe model ID (not typesafe: prefix)')
    parser.add_argument('--preprocessing', choices=['reference', 'raw'], default='reference',
                        help='raw uses full dialogue/candidates and entity choices, without BERT preprocessing')
    parser.add_argument('--data-dir', type=Path, default=Path('data/abcd'))
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--concurrency', type=int, default=4, help='Maximum in-flight API requests')
    args = parser.parse_args()
    if args.limit < 1 or args.concurrency < 1 or (args.example_limit is not None and args.example_limit < 1):
        parser.error('limits and concurrency must be positive')
    if args.model.startswith('typesafe:'):
        parser.error('--model expects a TypeSafe model ID, e.g. jev-1.13.0')
    if not args.dry_run and not os.environ.get('TYPESAFE_API_KEY'):
        parser.error('Set TYPESAFE_API_KEY or use --dry-run')
    config = BenchmarkConfig(
        **{k: v for k, v in vars(args).items() if k not in {'full_split', 'limit'}},
        limit=None if args.full_split else args.limit,
    )
    summary = asyncio.run(run_benchmark(config))
    print(json.dumps({k: v for k, v in summary.items() if k not in {'labels', 'confusions'}}, indent=2))


if __name__ == '__main__':
    main()
