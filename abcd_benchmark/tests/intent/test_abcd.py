import asyncio
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from pydantic_ai.usage import RunUsage
from typesafe_sdk import AsyncTypeSafeClient, ChoiceAnswer

from abcd_benchmark.common.data import read_json
from abcd_benchmark.intent.evaluation import (
    EvaluationConfig,
    predict_row,
    run_evaluation,
    select_conversations,
)
from abcd_benchmark.intent.model import Intent, IntentPrediction, dialogue_prefix
from abcd_benchmark.intent.summary import summarize


def test_prefix_excludes_labels_actions_and_future():
    conversation = {
        'convo_id': 1,
        'scenario': {'subflow': 'SECRET'},
        'delexed': [
            {'speaker': 'agent', 'text': 'Hello', 'targets': ['SECRET']},
            {'speaker': 'action', 'text': 'SECRET ACTION'},
            {'speaker': 'customer', 'text': 'I forgot my password', 'targets': ['SECRET']},
            {'speaker': 'agent', 'text': 'FUTURE'},
        ],
    }
    prefix = dialogue_prefix(conversation, 1)
    assert prefix[-1]['text'] == 'I forgot my password'
    assert 'SECRET' not in json.dumps(prefix)
    assert 'FUTURE' not in json.dumps(prefix)
    assert len(dialogue_prefix(conversation, 0)) == 3


def test_sampling_is_seeded_and_bounded():
    data = {'dev': list(range(100))}
    assert select_conversations(data, 'dev', 3, 42) == select_conversations(data, 'dev', 3, 42)
    assert len(select_conversations([1, 2], 'sample', 20, 42)) == 2
    with pytest.raises(ValueError):
        select_conversations([1], 'dev', 1, 42)


def test_metric_calculation():
    rows = [
        {'gold': 'a', 'predicted': 'a', 'usage': {'input_tokens': 10}, 'latency_seconds': 1},
        {'gold': 'b', 'predicted': 'a', 'usage': {'input_tokens': 20}, 'latency_seconds': 3},
    ]
    summary = summarize(rows, {'a': 'same_flow', 'b': 'same_flow'})
    assert summary['intent_accuracy'] == 0.5
    assert summary['flow_accuracy'] == 1
    assert summary['input_tokens'] == 30
    assert summary['per_intent']['a']['precision'] == 0.5
    assert summary['per_intent']['b']['recall'] == 0
    assert summary['macro_f1'] == pytest.approx(1 / 3)
    assert summary['estimated_cost_usd'] == pytest.approx(30 * 0.042 / 1_000_000)


def test_empty_summary():
    summary = summarize([], {'a': 'flow'})
    assert summary['intent_accuracy'] is None
    assert summary['macro_f1'] is None
    assert summary['input_tokens'] == 0


def test_full_split_retains_every_conversation():
    assert select_conversations({'test': [1, 2, 3]}, 'test', None, 42) == [1, 2, 3]


def test_output_and_score():
    result = SimpleNamespace(
        output=IntentPrediction(intent=Intent.recover_password),
        response=SimpleNamespace(
            provider_details={'confidence': {'intent': 0.9},
                              'probabilities': {'intent': {'recover_password': 0.95}}},
            model_name='jev-1.13.0',
        ),
        usage=RunUsage(input_tokens=100, cost=Decimal('0.0000042')),
    )
    agent = SimpleNamespace(run=AsyncMock(return_value=result))
    row = asyncio.run(predict_row(agent, {'state': {}, 'gold': 'recover_password'}))
    assert row['output'] == {'intent': 'recover_password'}
    assert row['score'] == 0.9
    assert row['predicted_probability'] == 0.95
    assert row['correct'] is True
    assert row['usage']['input_tokens'] == 100
    assert json.loads(json.dumps(row))['usage']['cost'] == '0.0000042'
    assert agent.run.call_args.args == ('{}',)


@pytest.fixture
def evaluation_data(monkeypatch):
    conversation = {
        'convo_id': 1, 'scenario': {'subflow': 'recover_password'},
        'delexed': [{'speaker': 'customer', 'text': 'Forgot password',
                    'targets': ['recover_password']}],
    }
    ontology = {'intents': {'subflows': {'flow': [i.value for i in Intent]}}}
    monkeypatch.setattr('abcd_benchmark.intent.evaluation.download', lambda *args: Path('fixture.json'))
    # Distinguish ontology and dataset reads without touching the network.
    artifacts = iter([ontology, {'test': [conversation] * 2}])
    monkeypatch.setattr('abcd_benchmark.intent.evaluation.read_json', lambda path: next(artifacts))


def test_partial_run_writes_summary_and_completed_rows(tmp_path, monkeypatch, evaluation_data):
    output = tmp_path / 'run.jsonl'
    completed = {'gold': 'recover_password', 'predicted': 'recover_password',
                 'usage': {'input_tokens': 100}, 'latency_seconds': 1, 'score': 0.9}
    monkeypatch.setattr('abcd_benchmark.intent.evaluation.make_agent', lambda model: object())
    monkeypatch.setattr('abcd_benchmark.intent.evaluation.predict_row',
                        AsyncMock(side_effect=[completed, RuntimeError('API failed')]))
    config = EvaluationConfig(output=output, split='test', limit=None, concurrency=2)
    with pytest.raises(RuntimeError, match='API failed'):
        asyncio.run(run_evaluation(config))
    assert len(output.read_text().splitlines()) == 1
    summary = json.loads(output.with_suffix('.summary.json').read_text())
    assert summary['complete'] is False
    assert summary['conversations'] == 1
    assert summary['estimated_cost_usd'] == pytest.approx(0.0000042)


def test_dry_run_full_split(tmp_path, evaluation_data):
    output = tmp_path / 'dry.jsonl'
    config = EvaluationConfig(output=output, split='test', limit=None, dry_run=True)
    summary = asyncio.run(run_evaluation(config))
    assert summary['complete'] is True
    assert summary['prepared_requests'] == 2
    assert len(output.read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        # Guard the existing output before any requests are issued.
        asyncio.run(run_evaluation(config))


def test_pydantic_ai_typesafe_compilation_and_confidence():
    # Exercise the real provider/agent and row serialization; mock only HTTP.
    client = AsyncTypeSafeClient(api_key='test-not-a-real-key')
    client.system_one = AsyncMock(return_value=SimpleNamespace(
        model='jev-1.13.0', request_id='test',
        usage=SimpleNamespace(input_tokens=100, output_tokens=0),
        answers={'intent': ChoiceAnswer(
            type='choice', choice='recover_password', confidence=0.9,
            probabilities={'recover_password': 1.0},
        )},
    ))
    provider = TypeSafeProvider(typesafe_client=client)
    agent = Agent(TypeSafeModel('jev-latest', provider=provider), output_type=IntentPrediction)
    row = asyncio.run(predict_row(agent, {
        'state': {'dialogue': [{'speaker': 'customer', 'text': 'Forgot password'}]},
        'gold': 'recover_password',
    }))
    assert row['output'] == {'intent': 'recover_password'}
    assert row['score'] == 0.9
    assert row['usage']['input_tokens'] == 100
    json.dumps(row)  # Include SDK-derived costs in the serialization regression test.
    questions = client.system_one.call_args.args[1]
    assert len(questions['intent'].criteria) == 55
    assert 'forgotten password' in str(questions['intent'].criteria['recover_password'])


@pytest.mark.skipif(not Path('data/abcd/abcd_sample.json').exists(), reason='Local data not downloaded')
def test_local_sample_and_ontology():
    data = read_json(Path('data/abcd/abcd_sample.json'))
    ontology = read_json(Path('data/abcd/ontology.json'))
    expected = {intent for intents in ontology['intents']['subflows'].values() for intent in intents}
    assert expected == {intent.value for intent in Intent}
    faq = next(c for c in data if c['scenario']['subflow'] == 'timing_4')
    assert faq['delexed'][0]['targets'][0] == 'timing'
    assert 'targets' not in json.dumps(dialogue_prefix(faq, 3))
