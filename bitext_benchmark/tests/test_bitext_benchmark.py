import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from typesafe_sdk import AsyncTypeSafeClient, ChoiceAnswer

from bitext_benchmark.data import load_data, normalize, select_rows, split_for
from bitext_benchmark.metrics import score
from bitext_benchmark.config import validate_model
from bitext_benchmark.model import predict
from bitext_benchmark.runner import run
from bitext_benchmark.schema import Prediction


def row(index=0, text='Please cancel my order', gold='cancel_order'):
    return {'id': f'row-{index:05d}', 'state': {'instruction': text},
            'gold': gold, 'category': 'ORDER', 'flags': 'BQ'}


def write_csv(path: Path, records: list[dict]):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['instruction', 'intent', 'category', 'flags', 'response'])
        writer.writeheader()
        for record in records:
            writer.writerow({'instruction': record['state']['instruction'], 'intent': record['gold'],
                             'category': record['category'], 'flags': record['flags'], 'response': 'SECRET_RESPONSE'})


def test_loading_and_no_response_leak(tmp_path):
    path = tmp_path / 'data.csv'
    write_csv(path, [row(), row(1, ' PLEASE CANCEL MY ORDER ')])
    rows, manifest = load_data(path)
    assert manifest['duplicate_rows'] == 1
    assert 'SECRET_RESPONSE' not in json.dumps(rows)
    assert manifest['source_url'] is None
    assert len(manifest['sha256']) == 64
    assert split_for(rows[0], 42) == split_for(rows[1], 42)


def test_conflicting_labels(tmp_path):
    path = tmp_path / 'data.csv'
    write_csv(path, [row(), row(1, gold='track_order')])
    with pytest.raises(ValueError, match='Conflicting'):
        load_data(path)


def test_bad_data(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('foo\nbar\n')
    with pytest.raises(ValueError, match='CSV must contain'):
        load_data(path)
    write_csv(path, [row(text=' ')])
    with pytest.raises(ValueError, match='Empty'):
        load_data(path)
    write_csv(path, [row(gold='unknown')])
    with pytest.raises(ValueError):
        load_data(path)


def test_split_determinism_and_disjointness():
    rows = [row(i, f'Message {i}') for i in range(200)]
    dev = select_rows(rows, split='dev', seed=42, limit=None)
    test = select_rows(rows, split='test', seed=42, limit=None)
    assert dev and test
    assert {r['id'] for r in dev}.isdisjoint(r['id'] for r in test)
    assert len(dev) + len(test) == len(rows)
    assert dev == select_rows(list(reversed(rows)), split='dev', seed=42, limit=None)
    assert normalize(' ＡBC  Test ') == 'abc test'
    with pytest.raises(ValueError):
        select_rows(rows, split='dev', seed=42, limit=0)


def test_pydanticai_prediction_and_metadata():
    agent = Agent(TestModel(custom_output_args={'intent': 'cancel_order'}), output_type=Prediction)
    result = asyncio.run(predict(agent, row()))
    assert result['correct']
    assert result['usage']['requests'] == 1
    assert result['latency_seconds'] >= 0


def test_real_jev_provider_compilation():
    client = AsyncTypeSafeClient(api_key='test-not-a-real-key')
    client.system_one = AsyncMock(return_value=SimpleNamespace(
        model='jev-1.13.0', request_id='test',
        usage=SimpleNamespace(input_tokens=123, output_tokens=0),
        answers={'intent': ChoiceAnswer(type='choice', choice='cancel_order', confidence=.9,
                                       probabilities={'cancel_order': 1.0})},
    ))
    agent = Agent(TypeSafeModel('jev-1.13.0', provider=TypeSafeProvider(typesafe_client=client)),
                  output_type=Prediction)
    result = asyncio.run(predict(agent, row()))
    assert result['confidence'] == .9
    assert result['probabilities'] == {'cancel_order': 1.0}
    assert result['usage']['input_tokens'] == 123
    state, questions = client.system_one.call_args.args
    assert 'Please cancel my order' in str(state)
    assert 'gold' not in str(state) and 'flags' not in str(state) and 'ORDER' not in str(state)
    assert len(questions['intent'].criteria) == 27
    assert 'existing order' in str(questions['intent'].criteria['cancel_order'])
    json.dumps(result)


def test_metrics():
    rows = [{**row(), 'predicted': 'cancel_order'},
            {**row(1, gold='track_order'), 'predicted': 'cancel_order'}]
    metrics = score(rows)
    assert metrics['accuracy'] == .5
    assert metrics['per_intent']['cancel_order']['f1'] == pytest.approx(2 / 3)
    assert metrics['macro_f1_supported'] == pytest.approx(1 / 3)
    assert metrics['flag_slices']['Q']['count'] == 2
    assert score([])['accuracy'] is None


def test_dry_run_and_exclusive_outputs(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError('Dry run constructed an agent')
    monkeypatch.setattr('bitext_benchmark.runner.make_agent', forbidden)
    output = tmp_path / 'inputs.jsonl'
    result = asyncio.run(run([row()], output=output, manifest={}, model='typesafe:jev-1.13.0', dry_run=True))
    assert result['metrics'] is None
    assert json.loads(output.read_text())['state'] == row()['state']
    with pytest.raises(FileExistsError):
        asyncio.run(run([row()], output=output, manifest={}, model='', dry_run=True))


def test_partial_summary(tmp_path, monkeypatch):
    async def fake_predict(agent, record):
        if record['id'] == 'row-00001':
            raise RuntimeError('API failure')
        return {**record, 'predicted': record['gold']}
    monkeypatch.setattr('bitext_benchmark.runner.predict', fake_predict)
    output = tmp_path / 'predictions.jsonl'
    with pytest.raises(RuntimeError):
        asyncio.run(run([row(), row(1)], output=output, manifest={}, model='', dry_run=False, agent=object()))
    summary = json.loads(output.with_suffix('.summary.json').read_text())
    assert summary['status'] == 'partial'
    assert summary['completed_count'] == 1
    assert len(output.read_text().splitlines()) == 1


def test_only_jev_allowed():
    validate_model('typesafe:jev-1.13.0')
    with pytest.raises(ValueError):
        validate_model('openai:gpt-4o')
