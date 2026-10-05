"""Score complete conversations from interrupted CDS JSONL runs, offline."""

import argparse
import json
from pathlib import Path

from abcd_benchmark.ast_cds.subsets import load_completed, summarize

__all__ = ['load_completed', 'main', 'summarize']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('data/abcd/abcd_v1.1.json.gz'))
    parser.add_argument('--ontology', type=Path, default=Path('data/abcd/ontology.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize(args.inputs, args.dataset, args.ontology)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for run in report['runs']:
        print(run['input'], run['complete_conversations'], 'complete conversations', run['metrics'])
    print('Paired complete conversations:', len(report['paired_complete_conversation_ids']))
    for run in report['paired_runs']:
        print(run['input'], 'paired', run['metrics'])


if __name__ == '__main__':
    main()
