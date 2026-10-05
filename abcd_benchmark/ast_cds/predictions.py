"""Pure conversion of normalized judgments into benchmark prediction rows."""

import numpy as np

from .contracts import JudgmentBatch
from .metrics import correctness, top_k
from .schema import Example, Labels, Task


def decode_prediction(example: Example, labels: Labels, task: Task,
                      judgments: JudgmentBatch, *, latency_seconds: float = 0.0) -> dict:
    """Apply reference argmax/tie rules and decode IDs without making API calls."""
    output_ids = {'intent': -1, 'nextstep': -1, 'action': 0, 'value': 0, 'utterance': 0}
    probabilities, scores = {}, {}
    utterance_probabilities = [0.0] * 100
    for name, head in judgments.heads.items():
        ordered = list(head.probabilities)
        probabilities[name] = ordered
        output_ids[name] = int(np.argmax(ordered))
        scores[name] = head.confidence
        if name == 'utterance':
            utterance_probabilities = ordered
            # Reference retrieval top-1 uses argpartition, including tie behavior.
            output_ids[name] = top_k(ordered, 1)[0]
    row = {
        **example.metadata(), 'gold_nextstep': labels.nextsteps[example.gold['nextstep']],
        'output_ids': output_ids,
        'output': {
            'action': labels.actions[output_ids['action']],
            'value': labels.values[output_ids['value']] if output_ids['value'] < len(labels.values)
                     else example.context_tokens[output_ids['value'] - len(labels.values)],
            'intent': labels.intents[output_ids['intent']] if task == 'cds' else None,
            'nextstep': labels.nextsteps[output_ids['nextstep']] if task == 'cds' else None,
            'utterance_index': output_ids['utterance'] if example.candidates else None,
            'utterance_id': example.candidates[output_ids['utterance']] if example.candidates else None,
        },
        'score': scores, 'probabilities': probabilities,
        'utterance_probabilities': utterance_probabilities,
        'resolved_models': list(judgments.resolved_models),
        'usage': {'input_tokens': judgments.input_tokens, 'output_tokens': judgments.output_tokens,
                  'requests': judgments.requests},
        'latency_seconds': latency_seconds,
    }
    row['correctness'] = correctness(row)
    return row
