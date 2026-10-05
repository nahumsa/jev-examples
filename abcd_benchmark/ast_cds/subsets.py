"""Offline completed-conversation coverage validation and paired CDS analysis."""

import json
from collections import defaultdict
from pathlib import Path

from abcd_benchmark.common.artifacts import artifact_hash
from abcd_benchmark.common.data import read_json

from .examples import build_examples
from .metrics import benchmark_metrics
from .schema import Labels


def load_completed(path: Path, expected: dict[int, list[str]]) -> tuple[dict[int, list[dict]], dict]:
    """Require exact example-ID coverage, not merely a terminal example."""
    saved = defaultdict(dict)
    protocols = set()
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get('dry_run') or row.get('task') != 'cds' or row.get('split') != 'test':
                raise ValueError(f'{path}:{number}: expected real test CDS predictions')
            convo_id, example_id = row['convo_id'], row['example_id']
            if convo_id not in expected or example_id not in expected[convo_id]:
                raise ValueError(f'{path}:{number}: unknown test example {example_id}')
            if example_id in saved[convo_id]:
                raise ValueError(f'{path}:{number}: duplicate example {example_id}')
            protocols.add(row['protocol'])
            # Exact request bodies are large and unnecessary for this calculation.
            row.pop('requests', None)
            saved[convo_id][example_id] = row
    completed = {
        convo_id: [rows[example_id] for example_id in expected[convo_id]]
        for convo_id, rows in saved.items() if set(rows) == set(expected[convo_id])
    }
    rows = [row for conversation in completed.values() for row in conversation]
    report = {
        'input': str(path), 'input_sha256': artifact_hash(path),
        'protocols': sorted(protocols), 'saved_examples': sum(len(r) for r in saved.values()),
        'complete_conversations': len(completed),
        'complete_conversation_ids': sorted(completed),
        'excluded_incomplete_conversations': sorted(set(saved) - set(completed)),
        'excluded_incomplete_examples': sum(len(saved[c]) for c in set(saved) - set(completed)),
        **benchmark_metrics(rows, 'cds'),
    }
    return completed, report


def summarize(inputs: list[Path], dataset: Path, ontology: Path) -> dict:
    """Use label-independent CDS expansion to determine expected ID coverage."""
    data = read_json(dataset)['test']
    labels = Labels.from_ontology(read_json(ontology))
    expected = defaultdict(list)
    # CDS retains all decisions regardless of reference value resolution. Raw
    # expansion has identical example IDs and requires no reference tokenizer.
    for example in build_examples(data, 'cds', labels, None, raw=True):
        expected[example.convo_id].append(example.example_id)
    completed_runs, reports = [], []
    for path in inputs:
        completed, report = load_completed(path, expected)
        completed_runs.append(completed)
        reports.append(report)
    shared = set.intersection(*(set(run) for run in completed_runs))
    paired = []
    for path, run in zip(inputs, completed_runs):
        rows = [row for convo_id in sorted(shared) for row in run[convo_id]]
        paired.append({'input': str(path), **benchmark_metrics(rows, 'cds')})
    return {
        'scope': 'Completed-conversation test subsets only; not full-test results',
        'method': 'Exact expected example-ID coverage, including all slots and the terminal decision; stable original ordering',
        'dataset': str(dataset), 'dataset_sha256': artifact_hash(dataset),
        'expected_test_conversations': len(expected),
        'expected_test_examples': sum(len(ids) for ids in expected.values()),
        'new_api_calls': 0,
        'runs': reports,
        'paired_complete_conversation_ids': sorted(shared),
        'paired_runs': paired,
    }
