"""Dependency-free closed-set metrics and language-tag diagnostics."""
from collections import Counter

from .schema import Intent


def score(rows: list[dict]) -> dict:
    correct = sum(r['gold'] == r['predicted'] for r in rows)
    per_intent = {}
    confusion = Counter((r['gold'], r['predicted']) for r in rows)
    for intent in Intent:
        label = intent.value
        support = sum(r['gold'] == label for r in rows)
        predicted = sum(r['predicted'] == label for r in rows)
        tp = confusion[label, label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_intent[label] = {'support': support, 'precision': precision, 'recall': recall, 'f1': f1}
    slices = {}
    for flag in sorted({flag for row in rows for flag in row['flags'] if flag.isalpha()}):
        subset = [r for r in rows if flag in r['flags']]
        slices[flag] = {'count': len(subset), 'accuracy': sum(r['gold'] == r['predicted'] for r in subset) / len(subset)}
    supported = [v['f1'] for v in per_intent.values() if v['support']]
    return {'count': len(rows), 'accuracy': correct / len(rows) if rows else None,
            'macro_f1_all_27': sum(v['f1'] for v in per_intent.values()) / len(Intent) if rows else None,
            'macro_f1_supported': sum(supported) / len(supported) if supported else None,
            'per_intent': per_intent,
            'confusion': [{'gold': a, 'predicted': b, 'count': n} for (a, b), n in sorted(confusion.items())],
            'flag_slices': slices,
            'usage': {key: sum(r.get('usage', {}).get(key, 0) or 0 for r in rows)
                      for key in ('input_tokens', 'output_tokens', 'requests')},
            'latency_seconds': sum(r.get('latency_seconds', 0) for r in rows)}
