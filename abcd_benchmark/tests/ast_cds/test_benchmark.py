import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from typesafe_sdk import ChoiceAnswer

from abcd_benchmark.ast_cds.examples import (
    TOKENIZER_REVISION,
    Labels,
    build_examples,
    copy_context,
    value_target,
)
from abcd_benchmark.ast_cds.metrics import (
    benchmark_metrics,
    cascading_score,
    correctness,
    top_k,
)
from abcd_benchmark.ast_cds.pipeline import (
    predict_example,
    prepare_requests,
    serialize_requests,
)
from abcd_benchmark.ast_cds.runner import BenchmarkConfig, run_benchmark


class FakeTokenizer:
    def tokenize(self, text):
        return text.lower().split()


@pytest.fixture
def ontology():
    return {
        'intents': {'subflows': {'flow': ['recover_password', 'recover_username']}},
        'next_steps': ['retrieve_utterance', 'take_action', 'end_conversation'],
        'actions': {'actions': {'verify-identity': ['customer_name', 'account_id', 'order_id'],
                               'reset': [], 'select': ['option']}},
        'values': {'enumerable': {'option': ['yes', 'no']},
                   'non_enumerable': {'entities': ['customer_name', 'account_id', 'order_id']}},
    }


@pytest.fixture
def labels(ontology):
    return Labels.from_ontology(ontology)


def turn(speaker, text, number, nextstep='retrieve_utterance', action='reset', values=None):
    return {'speaker': speaker, 'text': text, 'turn_count': number,
            'targets': ['recover_password', nextstep, action, values or [],
                        7 if nextstep == 'retrieve_utterance' else -1],
            'candidates': list(range(100)) if nextstep == 'retrieve_utterance' else []}


@pytest.fixture
def conversation():
    return {
        'convo_id': 12,
        'scenario': {'subflow': 'SECRET_SCENARIO'},
        'delexed': [
            turn('customer', 'Forgot my password. <customer_name> <account_id> <order_id>', 1),
            turn('agent', 'CURRENT_AGENT_SECRET', 2),
            turn('action', 'CURRENT_ACTION_SECRET', 3, 'take_action', 'verify-identity',
                 ['first', 'second', 'third']),
            turn('customer', 'Thanks FUTURE_SECRET', 4),
            turn('action', 'Password reset successful', 5, 'take_action'),
        ],
    }


def test_cds_expansion_boundaries_and_terminal(conversation, labels):
    examples = build_examples([conversation], 'cds', labels, FakeTokenizer())
    assert len(examples) == 6  # agent + three slots + reset + terminal
    assert [e.slot_index for e in examples[1:4]] == [0, 1, 2]
    assert len({e.example_id for e in examples}) == len(examples)
    assert 'current_agent_secret' not in examples[0].history_tokens
    assert 'current_agent_secret' in examples[1].history_tokens
    assert 'current_action_secret' not in examples[1].history_tokens
    assert 'future_secret' not in examples[1].history_tokens
    assert examples[1].context_tokens == examples[2].context_tokens == examples[3].context_tokens
    assert examples[1].gold['value'] == examples[2].gold['value'] == examples[3].gold['value']
    assert examples[-1].terminal
    assert examples[-1].gold['nextstep'] == 2
    assert examples[-1].gold['action'] == examples[-1].gold['value'] == examples[-1].gold['utterance'] == -1
    assert examples[-1].turn_count == examples[-2].turn_count  # preserve reference duplicate


def test_ast_expansion_action_names_and_missing_value_filter(conversation, labels):
    examples = build_examples([conversation], 'ast', labels, FakeTokenizer())
    assert len(examples) == 4
    assert 'verify-identity' in examples[-1].history_tokens
    assert 'current_action_secret' not in examples[-1].history_tokens
    assert examples[-1].gold['value'] == -1
    conversation['delexed'][0]['text'] = 'No entities here'
    examples = build_examples([conversation], 'ast', labels, FakeTokenizer())
    assert len(examples) == 1  # reference drops unavailable entity values in AST only
    assert len(build_examples([conversation], 'cds', labels, FakeTokenizer())) == 6


def test_history_keeps_first_tokens_and_copy_context_keeps_tail(labels):
    history = ['customer|' + ' '.join(f'token{i}' for i in range(600))]
    tokens = copy_context(history, 'verify-identity a', FakeTokenizer())
    assert len(tokens) == 95
    assert tokens[-1] == 'token599'
    conversation = {'convo_id': 1, 'delexed': [turn('customer', history[0].split('|')[1], 1),
                                              turn('action', 'CURRENT', 2, 'take_action')]}
    example = build_examples([conversation], 'ast', labels, FakeTokenizer())[0]
    assert len(example.history_tokens) == 510
    assert example.history_tokens[-1] == 'token509'


def test_reference_value_category_rule(labels):
    tokens = ['<account_id>', '<customer_name>', '<order_id>']
    assert value_target(tokens, 'verify-identity', 'ignored-value', labels) == len(labels.values) + 1
    assert value_target([], 'select', 'yes', labels) == 0
    assert value_target([], 'select', 'unknown', labels) == -1


def test_request_states_do_not_reveal_branch_or_gold(conversation, labels):
    examples = build_examples([conversation], 'cds', labels, FakeTokenizer())
    utterances = [f'candidate text {i}' for i in range(100)]
    requests = prepare_requests(examples[0], labels, 'cds', utterances, FakeTokenizer())
    assert set(requests[0]['questions']) == {'intent', 'nextstep', 'action'}
    assert set(requests[0]['state']) == {'history_tokens'}
    assert len(requests[2]['questions']['utterance'].criteria) == 100
    serialized = json.dumps(serialize_requests(requests))
    for secret in ['SECRET_SCENARIO', 'CURRENT_AGENT_SECRET', 'FUTURE_SECRET', 'gold', 'turn_count']:
        assert secret not in serialized
    action_requests = prepare_requests(examples[1], labels, 'cds', utterances, FakeTokenizer())
    assert set(action_requests[0]['state']) == {'history_tokens'}
    assert 'value_context_tokens' in action_requests[1]['state']
    assert len(action_requests[1]['questions']['value'].criteria) == len(labels.values) + len(examples[1].context_tokens)


def scored_row(number, branch='end_conversation', right=True, convo_id=1):
    step = ['retrieve_utterance', 'take_action', 'end_conversation'].index(branch)
    return {
        'convo_id': convo_id, 'turn_count': number,
        'gold': {'intent': 0, 'nextstep': step, 'action': 0 if step == 1 else -1,
                 'value': 0 if step == 1 else -1, 'utterance': 99 if step == 0 else -1},
        'gold_nextstep': branch,
        'output_ids': {'intent': 0 if right else 1, 'nextstep': step, 'action': 0, 'value': 0, 'utterance': 99},
        'utterance_probabilities': list(range(100)),
        'score': {'intent': 0.9}, 'usage': {'input_tokens': 10, 'output_tokens': 1, 'requests': 1},
        'latency_seconds': 1, 'resolved_models': ['test-model'],
    }


def test_paper_cascading_example():
    rows = [scored_row(i, right=i != 2) for i in range(4)]
    assert cascading_score(rows) == pytest.approx((2 / 4 + 1 / 3 + 0 + 1) / 4)
    summary = benchmark_metrics(rows, 'cds')
    assert summary['metrics']['Turn_Accuracy'] == 0.75
    assert summary['reference_rounded_metrics']['Cascading_Score'] == 0.4583


def test_cascade_weights_examples_and_preserves_duplicate_order():
    rows = [scored_row(1), scored_row(1, right=False), scored_row(2)]
    assert cascading_score(rows) == pytest.approx((1 / 3 + 0 + 1) / 3)
    rows += [scored_row(1, convo_id=2)]
    assert cascading_score(rows) == pytest.approx((1 / 3 + 0 + 1 + 1) / 4)


def test_metric_denominators_and_missing_value_quirk():
    rows = [scored_row(1, 'retrieve_utterance'), scored_row(2, 'take_action'),
            scored_row(3, 'take_action'), scored_row(4)]
    rows[2]['gold']['value'] = -1
    summary = benchmark_metrics(rows, 'cds')
    assert summary['denominators']['Action_Accuracy'] == 2
    assert summary['denominators']['Value_Accuracy'] == 1
    assert summary['denominators']['Recall_at_1'] == 1
    assert summary['metrics']['Value_Accuracy'] == 1
    assert summary['metrics']['Joint_Accuracy'] == 0.5
    assert summary['metrics']['Turn_Accuracy'] == 0.75
    assert correctness(rows[2])['turn'] is False
    ast = benchmark_metrics(rows[1:3], 'ast')
    assert ast['metrics'] == {'Bslot_Accuracy': 1, 'Value_Accuracy': 0.5, 'Joint_Accuracy': 0.5}
    assert benchmark_metrics([], 'cds')['metrics']['Cascading_Score'] is None


def test_ranking_ties_match_reference_numpy():
    probs = [0.0] * 100
    for k in [1, 5, 10]:
        assert top_k(probs, k) == np.argpartition(probs, kth=-k)[-k:].tolist()
    with pytest.raises(ValueError):
        top_k([0.1], 5)


def test_predict_example_real_sdk_answers_no_network(conversation, labels):
    example = build_examples([conversation], 'cds', labels, FakeTokenizer())[0]
    requests = prepare_requests(example, labels, 'cds', ['response'] * 100, FakeTokenizer())

    async def system_one(state, questions, *, model):
        answers = {key: ChoiceAnswer(choice='0', confidence=0.8,
                   probabilities={str(i): float(i == 0) for i in range(len(q.criteria))})
                   for key, q in questions.items()}
        return SimpleNamespace(model=model, answers=answers,
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))

    client = SimpleNamespace(system_one=AsyncMock(side_effect=system_one))
    row = asyncio.run(predict_example(client, example, labels, 'cds', requests, 'test-model', asyncio.Semaphore(1)))
    assert row['output']['intent'] == 'recover_password'
    assert row['output']['utterance_id'] == 0
    assert row['usage']['requests'] == 3
    assert row['usage']['input_tokens'] == 300
    assert row['score']['utterance'] == 0.8
    json.dumps(row)


@pytest.fixture
def runner_data(tmp_path, monkeypatch, ontology, conversation):
    from abcd_benchmark.ast_cds import runner
    for name, content in [('ontology.json', ontology), ('abcd_v1.1.json.gz', {'dev': [conversation]}),
                          ('utterances.json', ['response'] * 100)]:
        # Downloads are mocked; JSON payload is uncompressed for this fixture.
        (tmp_path / name.replace('.gz', '')).write_text(json.dumps(content))
    monkeypatch.setattr(runner, 'download', lambda name, directory: directory / name.replace('.gz', ''))
    monkeypatch.setattr(runner, 'load_tokenizer', lambda *args: FakeTokenizer())
    vocab = tmp_path / 'tokenizers' / TOKENIZER_REVISION / 'vocab.txt'
    vocab.parent.mkdir(parents=True)
    vocab.write_text('mock-vocabulary')
    return tmp_path


@pytest.mark.parametrize('task', ['ast', 'cds'])
def test_dry_run_outputs_and_exclusive_creation(task, runner_data):
    config = BenchmarkConfig(task=task, output=runner_data / f'{task}.jsonl', data_dir=runner_data, dry_run=True, limit=None)
    summary = asyncio.run(run_benchmark(config))
    assert summary['complete'] is True
    assert summary['full_split_evaluation'] is True
    assert summary['prepared_examples'] == (4 if task == 'ast' else 6)
    rows = [json.loads(line) for line in config.output.read_text().splitlines()]
    assert rows[0]['requests'][0]['state']['history_tokens']
    assert json.loads(config.output.with_suffix('.summary.json').read_text()) == summary
    with pytest.raises(FileExistsError):
        asyncio.run(run_benchmark(config))


def test_partial_run_preserves_success_and_suppresses_cascade(runner_data, monkeypatch):
    from abcd_benchmark.ast_cds import runner

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    async def predict(client, example, *args):
        if example.slot_index == 1:
            raise RuntimeError('HTTP failed')
        return {**scored_row(example.turn_count), **example.metadata(),
                'gold_nextstep': 'retrieve_utterance' if example.gold['nextstep'] == 0 else 'take_action'}

    monkeypatch.setattr(runner, 'AsyncTypeSafeClient', lambda **kwargs: Client())
    monkeypatch.setattr(runner, 'predict_example', predict)
    config = BenchmarkConfig(task='cds', output=runner_data / 'partial.jsonl', data_dir=runner_data, concurrency=4)
    with pytest.raises(RuntimeError, match='HTTP failed'):
        asyncio.run(run_benchmark(config))
    summary = json.loads(config.output.with_suffix('.summary.json').read_text())
    assert summary['complete'] is False
    assert summary['examples'] == 3
    assert summary['metrics']['Cascading_Score'] is None
    assert summary['failed_examples'][0]['example_id'] == '12:3:1:0'


@pytest.mark.parametrize('task', ['ast', 'cds'])
def test_raw_uses_full_text_and_never_tokenizes(task, conversation, labels):
    conversation['delexed'][0]['text'] += ' ' + ' '.join(f'WORD{i}' for i in range(600))
    examples = build_examples([conversation], task, labels, None, raw=True)
    example = examples[0]
    assert example.history_tokens == []
    assert 'WORD599' in example.history_text
    assert 'CURRENT_ACTION_SECRET' not in example.history_text
    if task == 'cds':
        assert 'CURRENT_AGENT_SECRET' not in example.history_text
    assert example.context_tokens == ['<customer_name>', '<account_id>', '<order_id>']
    candidates = ['RAW RESPONSE ' + ' '.join(f'WORD{i}' for i in range(600))] * 100
    requests = prepare_requests(example, labels, task, candidates, None)
    assert requests[0]['state'] == {'history': example.history_text}
    assert 'history_tokens' not in json.dumps(requests[0]['state'])
    assert '`history`' in requests[0]['questions']['action'].instructions
    if task == 'cds':
        assert requests[2]['questions']['utterance'].criteria['0'] == candidates[0]


@pytest.mark.parametrize('task', ['ast', 'cds'])
def test_raw_runner_does_not_load_or_hash_vocabulary(task, runner_data, monkeypatch):
    def forbidden(*args):
        raise AssertionError('Raw evaluation must not load a reference tokenizer')

    monkeypatch.setattr('abcd_benchmark.ast_cds.runner.load_tokenizer', forbidden)
    config = BenchmarkConfig(task=task, output=runner_data / f'raw-{task}.jsonl',
                             data_dir=runner_data, preprocessing='raw', dry_run=True)
    summary = asyncio.run(run_benchmark(config))
    assert summary['complete']
    assert summary['tokenizer'] is summary['tokenizer_revision'] is None
    assert 'tokenizer_vocab' not in summary['source_sha256']
    assert summary['protocol'] == 'abcd-v1.1-raw-full-history-entities'


def test_raw_entity_choices_do_not_depend_on_gold_action(conversation, labels):
    first = build_examples([conversation], 'cds', labels, None, raw=True)[1]
    conversation['delexed'][2]['targets'][2] = 'select'
    conversation['delexed'][2]['targets'][3] = ['yes']
    second = build_examples([conversation], 'cds', labels, None, raw=True)[1]
    assert first.context_tokens == second.context_tokens
    assert first.history_text == second.history_text
