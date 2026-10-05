"""Compare against pinned upstream code, not a second rewrite of the scorer."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abcd_benchmark.ast_cds.examples import Labels, build_examples
from abcd_benchmark.ast_cds.metrics import benchmark_metrics

SOURCE_DIR = Path(__file__).parent / 'reference_sources'


def reference_namespace(filename: str, names: set[str]) -> dict:
    source = ast.parse((SOURCE_DIR / filename).read_text())
    selected = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {'np': np, 'progress_bar': lambda data, **kwargs: data}
    # Execute only checked-in, trusted upstream test fixtures; never user/network input.
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)  # noqa: S102
    return namespace


def reference_examples(conversations, ontology, tokenizer, task):
    namespace = reference_namespace('process.py.txt', {
        'prepare_action_labels', 'prepare_intent_labels', 'prepare_nextstep_labels',
        'prepare_value_labels', 'BaseProcessor', 'ASTProcessor', 'CDSProcessor',
    })
    base = namespace['BaseProcessor']
    # Token padding/embeddings are not needed to inspect reference labels/context.
    base.prepare_special_tokens = lambda self, args: None

    def record(self, history, target_ids, context_tokens, intent=None, candidates=None):
        return {'history': history, 'gold': target_ids, 'context_tokens': context_tokens,
                'candidates': candidates}

    base.convert_example = record
    args = SimpleNamespace(task=task, model_type='bert', use_intent=False)
    processor = namespace['ASTProcessor' if task == 'ast' else 'CDSProcessor'](args, tokenizer, ontology)
    return processor.build_features(args, {'test': conversations})['test']


class Tokenizer:
    def tokenize(self, text):
        return text.lower().split()


def fixture_data():
    ontology = {
        'intents': {'subflows': {'flow': ['intent_a', 'intent_b']}},
        'next_steps': ['retrieve_utterance', 'take_action', 'end_conversation'],
        'actions': {'section': {'verify-identity': ['customer_name', 'account_id', 'order_id'],
                               'select': ['option'], 'reset': []}},
        'values': {'enumerable': {'option': ['yes', 'no']},
                   'non_enumerable': {'entities': ['customer_name', 'account_id', 'order_id']}},
    }

    def turn(speaker, text, step, action='reset', values=None):
        return {'speaker': speaker, 'text': text, 'turn_count': 0,
                'targets': ['intent_a', step, action, values or [], 7 if step == 'retrieve_utterance' else -1],
                'candidates': list(range(100)) if step == 'retrieve_utterance' else []}

    conversations = []
    for index in range(3):
        turns = [
            turn('customer', 'hello <customer_name> <account_id> <order_id>' if index else 'no placeholders', 'retrieve_utterance'),
            turn('agent', 'May I verify your identity?', 'retrieve_utterance'),
            turn('action', 'Identity confirmed', 'take_action', 'verify-identity', ['name', 'id', 'order']),
            turn('customer', 'yes ' + ' '.join(f'word{i}' for i in range(index * 300)), 'retrieve_utterance'),
            turn('action', 'selected option', 'take_action', 'select', ['yes' if index else 'missing']),
            turn('action', 'reset completed', 'take_action'),
            turn('agent', 'Goodbye', 'retrieve_utterance'),
        ]
        for count, item in enumerate(turns, 1):
            item['turn_count'] = count
        conversations.append({'convo_id': index, 'delexed': turns})
    return ontology, conversations


@pytest.mark.parametrize('task', ['ast', 'cds'])
def test_preprocessing_matches_actual_reference(task):
    ontology, conversations = fixture_data()
    tokenizer = Tokenizer()
    labels = Labels.from_ontology(ontology)
    expected = reference_examples(conversations, ontology, tokenizer, task)
    actual = build_examples(conversations, task, labels, tokenizer)
    assert len(actual) == len(expected)
    for ours, upstream in zip(actual, expected):
        assert ours.context_tokens == upstream['context_tokens']
        history = ' [SEP] '.join(u.split('|')[1] for u in upstream['history'])
        assert ours.history_tokens == tokenizer.tokenize(history or '[PAD]')[:510]
        for field in ['action', 'value'] + (['intent', 'nextstep', 'utterance'] if task == 'cds' else []):
            assert ours.gold[field] == upstream['gold'][field]
        if task == 'cds':
            assert ours.convo_id == upstream['gold']['convo']
            assert ours.turn_count == upstream['gold']['turn']
            assert ours.candidates == ([] if upstream['candidates'] == [-1] * 100 else upstream['candidates'])


class Tensor:
    """Just the three methods used by cds_report to access IDs."""
    def __init__(self, data):
        self.data = np.array(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


@pytest.mark.parametrize('seed', [0, 7, 42])
def test_metrics_match_actual_reference(seed):
    namespace = reference_namespace('evaluate.py.txt', {'ast_report', 'cds_report'})
    rng = np.random.default_rng(seed)
    n = 30
    sizes = [2, 3, 3, 6, 100]
    predictions = [rng.random((n, size)) for size in sizes]
    # Include tied scores to preserve numpy's reference top-k behavior.
    predictions[-1][0] = 0
    predictions[-1][1] = 0.01
    labels = [rng.integers(size, size=n) for size in sizes]
    labels[1] = np.tile([0, 1, 2], n // 3)
    labels[2][labels[1] != 1] = -1
    labels[3][labels[1] != 1] = -1
    labels[3][4] = -1  # no-value action
    labels[4][labels[1] != 0] = -1
    # Duplicate turn counts deliberately exercise multi-slot stable ordering.
    convo_ids = np.repeat([0, 1, 2], 10)
    turn_counts = np.tile([1, 2, 3, 3, 3, 4, 5, 6, 7, 7], 3)
    rows = []
    fields = ['intent', 'nextstep', 'action', 'value', 'utterance']
    for index in range(n):
        rows.append({
            'convo_id': int(convo_ids[index]), 'turn_count': int(turn_counts[index]),
            'gold': {field: int(values[index]) for field, values in zip(fields, labels)},
            'output_ids': {field: int(np.argmax(scores[index])) for field, scores in zip(fields, predictions)},
            'gold_nextstep': ['retrieve_utterance', 'take_action', 'end_conversation'][labels[1][index]],
            'utterance_probabilities': predictions[-1][index].tolist(), 'score': {},
            'usage': {'input_tokens': 0, 'output_tokens': 0, 'requests': 0},
            'latency_seconds': 0, 'resolved_models': [],
        })
    reference, _ = namespace['cds_report'](predictions, labels, (Tensor(convo_ids), Tensor(turn_counts)))
    summary = benchmark_metrics(rows, 'cds')
    assert summary['reference_rounded_metrics'] == reference
    actual = summary['metrics']
    for field, value in reference.items():
        # Upstream rounds most metrics to four decimals, but leaves recall unrounded.
        assert actual[field] == pytest.approx(value, abs=0.00005)
    reference_ast, _ = namespace['ast_report'](predictions[2:4], labels[2:4])
    actual_ast = benchmark_metrics(rows, 'ast')['reference_rounded_metrics']
    assert actual_ast == reference_ast
