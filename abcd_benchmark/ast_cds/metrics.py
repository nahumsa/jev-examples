"""Pure equivalents of ABCD ast_report/cds_report (without KB masking).

Missing labels (-1) never match a nonnegative argmax prediction. This includes
no-value actions in Joint_Accuracy and cascading correctness: intentionally
reference-compatible, not a corrected variant of the task.
"""

from collections import Counter, defaultdict

import numpy as np

from abcd_benchmark.intent.metrics import INPUT_PRICE_PER_MILLION_USD

from .schema import Task


def ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def top_k(probabilities: list[float], k: int) -> list[int]:
    """Match the reference numpy argpartition tie behavior exactly."""
    if len(probabilities) < k:
        raise ValueError(f'Recall@{k} requires at least {k} candidates')
    return np.argpartition(probabilities, kth=-k)[-k:].tolist()


def correctness(row: dict) -> dict[str, bool]:
    gold, predicted = row['gold'], row['output_ids']
    match = {key: gold[key] >= 0 and gold[key] == predicted.get(key) for key in ['intent', 'nextstep', 'action', 'value']}
    match['joint'] = match['action'] and match['value']
    match['utterance'] = gold['utterance'] >= 0 and gold['utterance'] in top_k(row['utterance_probabilities'], 1)
    branch = row['gold_nextstep']
    realization = match['utterance'] if branch == 'retrieve_utterance' else match['joint'] if branch == 'take_action' else True
    match['turn'] = match['intent'] and match['nextstep'] and realization
    return match


def cascading_score(rows: list[dict]) -> float | None:
    """Average consecutive-success/remaining-length over all example starts.

    Stable sort retains upstream multi-slot and terminal duplicates at the same
    turn count. Longer conversations contribute more examples, just as upstream.
    """
    conversations = defaultdict(list)
    for row in rows:
        conversations[row['convo_id']].append(row)
    total = 0.0
    for conversation in conversations.values():
        ordered = sorted(conversation, key=lambda row: row['turn_count'])
        streak = 0
        for index in range(len(ordered) - 1, -1, -1):
            streak = streak + 1 if correctness(ordered[index])['turn'] else 0
            total += streak / (len(ordered) - index)
    return ratio(total, len(rows))


def benchmark_metrics(rows: list[dict], task: Task) -> dict:
    """Return full-precision scores, reference-rounded scores, and denominators."""
    matches = [correctness(row) for row in rows]
    n = len(rows)
    if task == 'ast':
        denominators = {'Bslot_Accuracy': n, 'Value_Accuracy': n, 'Joint_Accuracy': n}
        metrics = {
            'Bslot_Accuracy': ratio(sum(m['action'] for m in matches), n),
            'Value_Accuracy': ratio(sum(m['value'] for m in matches), n),
            'Joint_Accuracy': ratio(sum(m['joint'] for m in matches), n),
        }
    else:
        action_n = sum(row['gold']['action'] >= 0 for row in rows)
        value_n = sum(row['gold']['value'] >= 0 for row in rows)
        utterance_n = sum(row['gold']['utterance'] >= 0 for row in rows)
        denominators = {
            'Intent_Accuracy': n, 'Nextstep_Accuracy': n, 'Action_Accuracy': action_n,
            'Value_Accuracy': value_n, 'Joint_Accuracy': action_n,
            'Recall_at_1': utterance_n, 'Recall_at_5': utterance_n, 'Recall_at_10': utterance_n,
            'Turn_Accuracy': n, 'Cascading_Score': n,
        }
        metrics = {
            key: ratio(sum(m[match] for m in matches), denominators[key])
            for key, match in [('Intent_Accuracy', 'intent'), ('Nextstep_Accuracy', 'nextstep'),
                               ('Action_Accuracy', 'action'), ('Value_Accuracy', 'value'),
                               ('Joint_Accuracy', 'joint'), ('Turn_Accuracy', 'turn')]
        }
        for k in [1, 5, 10]:
            count = sum(row['gold']['utterance'] >= 0 and row['gold']['utterance'] in top_k(row['utterance_probabilities'], k) for row in rows)
            metrics[f'Recall_at_{k}'] = ratio(count, utterance_n)
        metrics['Cascading_Score'] = cascading_score(rows)
    input_tokens = sum(row['usage']['input_tokens'] for row in rows)
    confidences = defaultdict(list)
    for row in rows:
        for key, value in row['score'].items():
            confidences[key].append(value)
    confusions = {}
    for field in ['intent', 'nextstep', 'action', 'value']:
        pairs = Counter((row['gold'][field], row['output_ids'][field]) for row in rows
                        if row['gold'][field] >= 0 and row['gold'][field] != row['output_ids'][field])
        confusions[field] = [{'gold_id': a, 'predicted_id': b, 'count': count}
                             for (a, b), count in pairs.most_common()]
    return {
        'examples': n, 'conversations': len({row['convo_id'] for row in rows}),
        'metrics': metrics,
        # Upstream rounds accuracy/cascade, but returns retrieval recall unrounded.
        'reference_rounded_metrics': {
            k: (v if k.startswith('Recall_at_') else round(v, 4)) if v is not None else None
            for k, v in metrics.items()
        },
        'denominators': denominators,
        'no_value_action_examples': sum(row['gold']['action'] >= 0 and row['gold']['value'] < 0 for row in rows),
        'confusions': confusions,
        'mean_score': {k: sum(v) / len(v) for k, v in confidences.items()},
        'score_definition': 'Per-question Jev Choice confidence, not observed correctness',
        'input_tokens': input_tokens,
        'output_tokens': sum(row['usage']['output_tokens'] for row in rows),
        'requests': sum(row['usage']['requests'] for row in rows),
        'estimated_cost_usd': input_tokens * INPUT_PRICE_PER_MILLION_USD / 1_000_000,
        'input_price_per_million_usd': INPUT_PRICE_PER_MILLION_USD,
        'mean_latency_seconds': ratio(sum(row['latency_seconds'] for row in rows), n),
        'resolved_models': sorted({model for row in rows for model in row['resolved_models']}),
    }
