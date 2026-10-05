import json

import pytest

from abcd_benchmark.ast_cds.subsets import load_completed


def row(convo_id, number, right=True):
    return {
        'task': 'cds', 'split': 'test', 'dry_run': False, 'protocol': 'test',
        'convo_id': convo_id, 'turn_count': number, 'example_id': f'{convo_id}:{number}',
        'gold': {'intent': 0, 'nextstep': 2, 'action': -1, 'value': -1, 'utterance': -1},
        'gold_nextstep': 'end_conversation',
        'output_ids': {'intent': 0 if right else 1, 'nextstep': 2, 'action': 0, 'value': 0},
        'utterance_probabilities': [0.0] * 100, 'score': {},
        'usage': {'input_tokens': 0, 'output_tokens': 0, 'requests': 0},
        'latency_seconds': 0, 'resolved_models': ['test'],
    }


def test_subset_excludes_incomplete_conversation_even_with_terminal(tmp_path):
    expected = {1: ['1:1', '1:2', '1:3', '1:4'], 2: ['2:1', '2:2', '2:3']}
    rows = [row(1, n, right=n != 3) for n in range(1, 5)]
    rows += [row(2, 1), row(2, 3)]  # Last decision exists, but the middle is missing.
    path = tmp_path / 'predictions.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    completed, summary = load_completed(path, expected)
    assert list(completed) == [1]
    assert summary['saved_examples'] == 6
    assert summary['examples'] == 4
    assert summary['excluded_incomplete_conversations'] == [2]
    assert summary['excluded_incomplete_examples'] == 2
    assert summary['metrics']['Cascading_Score'] == pytest.approx((2 / 4 + 1 / 3 + 0 + 1) / 4)


def test_subset_rejects_duplicate_predictions(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    prediction = row(1, 1)
    path.write_text((json.dumps(prediction) + '\n') * 2)
    with pytest.raises(ValueError, match='duplicate example'):
        load_completed(path, {1: ['1:1']})


def test_subset_uses_expected_order_not_file_order(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    rows = [row(1, 2, right=False), row(1, 1)]
    for r in rows:
        r['turn_count'] = 1  # Duplicate turn count requires stable expansion order.
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    completed, report = load_completed(path, {1: ['1:1', '1:2']})
    assert [r['example_id'] for r in completed[1]] == ['1:1', '1:2']
    assert report['metrics']['Cascading_Score'] == 0.25
