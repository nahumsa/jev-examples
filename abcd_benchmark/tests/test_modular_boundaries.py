import asyncio
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from abcd_benchmark.ast_cds.config import BenchmarkConfig
from abcd_benchmark.ast_cds.datasets import PreparedBenchmark, load_benchmark_data
from abcd_benchmark.ast_cds.reporting import build_summary
from abcd_benchmark.ast_cds.schema import Example, Labels
from abcd_benchmark.common.artifacts import artifact_hash, ensure_new_artifacts
from abcd_benchmark.common.data import read_json
from abcd_benchmark.common.selection import select_conversations
from abcd_benchmark.intent.config import EvaluationConfig
from abcd_benchmark.intent.runner import run_evaluation
from abcd_benchmark.intent.schema import Intent


def test_existing_imports_refer_to_canonical_definitions():
    from abcd_benchmark.ast_cds import examples, runner
    from abcd_benchmark.intent import evaluation as abcd_evaluation
    from abcd_benchmark.intent import metrics
    from abcd_benchmark.intent import model as abcd_jev
    from abcd_benchmark.intent import summary as abcd_metrics

    assert abcd_evaluation.EvaluationConfig is EvaluationConfig
    assert abcd_evaluation.select_conversations is select_conversations
    assert abcd_jev.Intent is Intent
    assert abcd_metrics.summarize is metrics.summarize
    assert examples.Example is Example
    assert examples.Labels is Labels
    assert runner.BenchmarkConfig is BenchmarkConfig
    assert runner.artifact_hash is artifact_hash


def test_offline_modules_do_not_import_inference_dependencies():
    # Separate process prevents prior tests' imports from hiding coupling.
    script = '''
import sys
import abcd_benchmark.common.selection
import abcd_benchmark.intent.requests
import abcd_benchmark.intent.metrics
import abcd_benchmark.ast_cds.schema
import abcd_benchmark.ast_cds.preprocessing
import abcd_benchmark.ast_cds.subsets
assert 'pydantic_ai' not in sys.modules
assert 'typesafe_sdk' not in sys.modules
assert 'transformers' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=20)


def test_intent_runner_accepts_offline_dependencies(tmp_path):
    ontology = {'intents': {'subflows': {'flow': [i.value for i in Intent]}}}
    conversation = {'convo_id': 1, 'scenario': {'subflow': 'recover_password'},
                    'delexed': [{'speaker': 'customer', 'text': 'Forgot password',
                                 'targets': ['recover_password']}]}
    for name, data in [('ontology.json', ontology), ('dataset.json', {'test': [conversation]})]:
        (tmp_path / name).write_text(json.dumps(data))

    def forbidden(model):
        raise AssertionError('Dry runs must not construct an inference agent')

    config = EvaluationConfig(split='test', limit=None, dry_run=True, data_dir=tmp_path,
                              dataset=tmp_path / 'dataset.json', output=tmp_path / 'intent.jsonl')
    summary = asyncio.run(run_evaluation(config, download_artifact=lambda name, directory: directory / name,
                                        load_json=read_json, agent_factory=forbidden))
    assert summary['prepared_requests'] == 1
    assert summary['complete'] is True
    row = json.loads(config.output.read_text())
    assert row['state'] == {'dialogue': [{'speaker': 'customer', 'text': 'Forgot password'}]}


def test_benchmark_loader_exposes_scope_without_inference_or_tokenization(tmp_path):
    ontology = {
        'intents': {'subflows': {'flow': ['a', 'b']}},
        'next_steps': ['retrieve_utterance', 'take_action', 'end_conversation'],
        'actions': {'actions': {'act1': [], 'act2': []}},
        'values': {'enumerable': {'option': ['yes', 'no']}, 'non_enumerable': {}},
    }
    conversation = {'convo_id': 1, 'delexed': [
        {'speaker': 'customer', 'text': 'Hello', 'turn_count': 1,
         'targets': ['a', 'retrieve_utterance', 'act1', [], 0], 'candidates': list(range(100))},
        {'speaker': 'agent', 'text': 'Hi', 'turn_count': 2,
         'targets': ['a', 'retrieve_utterance', 'act1', [], 0], 'candidates': list(range(100))},
    ]}
    for name, data in [('ontology.json', ontology), ('utterances.json', ['Hi'] * 100),
                       ('dataset.json', {'test': [conversation]})]:
        (tmp_path / name).write_text(json.dumps(data))

    def forbidden(*args):
        raise AssertionError('Raw preparation must not load a tokenizer')

    config = BenchmarkConfig(task='cds', split='test', limit=None, preprocessing='raw', example_limit=1,
                             data_dir=tmp_path, dataset=tmp_path / 'dataset.json', output=tmp_path / 'unused.jsonl')
    prepared = load_benchmark_data(config, download_artifact=lambda name, directory: directory / name,
                                   load_json=read_json, tokenizer_factory=forbidden)
    assert prepared.total_selected_examples == 2
    assert len(prepared.examples) == 1
    assert prepared.selected_conversations == 1
    assert prepared.full_split_evaluation is False
    assert prepared.tokenizer is None
    assert 'tokenizer_vocab' not in prepared.source_hashes
    assert not config.output.exists()


@pytest.mark.parametrize('complete,limited,dry_run', [(True, False, False), (False, False, False),
                                                     (True, True, False), (True, False, True)])
def test_reporting_owns_cascading_validity(complete, limited, dry_run, tmp_path):
    example = Example(example_id='1:1:0:1', convo_id=1, turn_count=1, slot_index=0, terminal=True,
                      history_tokens=[], context_tokens=[], candidates=[],
                      gold={'intent': 0, 'nextstep': 2, 'action': -1, 'value': -1, 'utterance': -1})
    labels = Labels(intents=['a', 'b'], actions=['act1', 'act2'], values=['yes', 'no'],
                    nextsteps=['retrieve_utterance', 'take_action', 'end_conversation'],
                    value_by_action={}, enumerable={})
    prepared = PreparedBenchmark(protocol='test', labels=labels, tokenizer=None, utterances=[],
                                 examples=[example], source_hashes={}, selected_conversations=1,
                                 total_selected_examples=1, full_split_evaluation=True)
    if limited:
        prepared = replace(prepared, total_selected_examples=2, full_split_evaluation=False)
    config = BenchmarkConfig(task='cds', output=tmp_path / 'unused.jsonl', dry_run=dry_run)
    row = {**example.metadata(), 'gold_nextstep': 'end_conversation',
           'output_ids': {'intent': 0, 'nextstep': 2, 'action': 0, 'value': 0},
           'utterance_probabilities': [0.0] * 100, 'score': {},
           'usage': {'input_tokens': 0, 'output_tokens': 0, 'requests': 2},
           'latency_seconds': 0, 'resolved_models': ['fake']}
    summary = build_summary(config, prepared, [row], complete=complete, failed_examples=[],
                            prepared_request_count=2, wall_time_seconds=0)
    valid = complete and not limited and not dry_run
    assert summary['cascading_score_valid_for_selected_conversations'] is valid
    if dry_run:
        assert 'metrics' not in summary
        assert summary['prepared_requests'] == 2
    else:
        assert summary['metrics']['Cascading_Score'] == (1 if valid else None)


@pytest.mark.parametrize('module', ['abcd_benchmark', 'abcd_benchmark.cli.benchmark', 'abcd_benchmark.cli.intent',
                                   'abcd_benchmark.cli.download', 'abcd_benchmark.cli.summarize_cds'])
def test_package_cli_help(module):
    result = subprocess.run([sys.executable, '-m', module, '--help'], check=True,
                            capture_output=True, text=True, timeout=20)
    assert 'usage:' in result.stdout


def test_artifact_guard_rejects_an_existing_summary(tmp_path):
    output = tmp_path / 'predictions.jsonl'
    output.with_suffix('.summary.json').write_text('{}')
    with pytest.raises(FileExistsError):
        ensure_new_artifacts(output)
    assert not output.exists()
