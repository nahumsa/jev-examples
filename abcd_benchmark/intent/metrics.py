"""Pure evaluation metrics for ABCD intent predictions."""

from collections import Counter
from statistics import mean

INPUT_PRICE_PER_MILLION_USD = 0.042


def summarize(rows: list[dict], intent_to_flow: dict[str, str]) -> dict:
    """Summarize correctness, confidence, usage, latency, and all confusions."""
    correct = sum(row['predicted'] == row['gold'] for row in rows)
    flow_correct = sum(
        intent_to_flow[row['predicted']] == intent_to_flow[row['gold']] for row in rows
    )
    confusions = Counter(
        (row['gold'], row['predicted']) for row in rows if row['gold'] != row['predicted']
    )
    input_tokens = sum(row['usage']['input_tokens'] for row in rows)
    per_intent = {}
    for intent in sorted(intent_to_flow):
        support = sum(row['gold'] == intent for row in rows)
        predicted = sum(row['predicted'] == intent for row in rows)
        tp = sum(row['gold'] == row['predicted'] == intent for row in rows)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_intent[intent] = {
            'support': support, 'predicted': predicted, 'correct': tp,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    scores = [row['score'] for row in rows if row.get('score') is not None]
    return {
        'conversations': len(rows), 'correct': correct, 'incorrect': len(rows) - correct,
        'intent_accuracy': correct / len(rows) if rows else None,
        'flow_accuracy': flow_correct / len(rows) if rows else None,
        'macro_f1': mean(m['f1'] for m in per_intent.values() if m['support']) if rows else None,
        'mean_score': mean(scores) if scores else None,
        'score_definition': 'Jev Choice confidence (0–1), not correctness or a Score primitive',
        'mean_latency_seconds': mean(row['latency_seconds'] for row in rows) if rows else None,
        'input_tokens': input_tokens,
        'output_tokens': sum(row['usage'].get('output_tokens', 0) for row in rows),
        'requests': sum(row['usage'].get('requests', 1) for row in rows),
        'estimated_cost_usd': input_tokens * INPUT_PRICE_PER_MILLION_USD / 1_000_000,
        'input_price_per_million_usd': INPUT_PRICE_PER_MILLION_USD,
        'resolved_models': sorted({row['resolved_model'] for row in rows if 'resolved_model' in row}),
        'per_intent': per_intent,
        'confusions': [
            {'gold': gold, 'predicted': predicted, 'count': count}
            for (gold, predicted), count in confusions.most_common()
        ],
    }
