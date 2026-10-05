"""Build independent benchmark-head requests without network or file I/O.

Action/intent/next-step heads must never see response pools or copy context:
these support inputs can reveal the annotated next-step branch. SDK question
objects are retained for compatibility with the existing request artifacts.
"""

from typing import TypedDict

from typesafe_sdk import Choice, JSONValue

from .schema import Example, Labels, Task, Tokenizer


class JudgmentRequest(TypedDict):
    state: dict[str, JSONValue]
    questions: dict[str, Choice]


def choice(options: list[str], instructions: str) -> Choice:
    if not 2 <= len(options) <= 255:
        raise ValueError(f'Jev Choice requires 2–255 options, got {len(options)}')
    return Choice(instructions=instructions, criteria={str(i): text for i, text in enumerate(options)})


def prepare_requests(example: Example, labels: Labels, task: Task,
                     utterances: list[str], tokenizer: Tokenizer | None,
                     utterance_cache: dict[int, str] | None = None) -> list[JudgmentRequest]:
    """Return exactly the request states/questions; labels remain outside them."""
    raw = example.history_text is not None
    history: dict[str, JSONValue] = {'history': example.history_text} if raw else {'history_tokens': example.history_tokens}
    questions = {
        'action': choice(labels.actions,
                         'Assuming the next agent step is an action, which ABCD button should be used next? '
                         'Infer the primary request and procedural progress from `history_tokens`. '
                         'Do not simply repeat a completed action.'),
    }
    if task == 'cds':
        questions.update(
            intent=choice(labels.intents, 'Which ABCD intent explains the primary customer request in `history_tokens`? '
                          'Distinguish the primary request from verification and procedural follow-ups.'),
            nextstep=choice(labels.nextsteps, 'What should the agent do immediately after `history_tokens`: '
                            'speak to the customer, execute a backend action, or end a completed conversation?'),
        )
    requests: list[JudgmentRequest] = [{'state': history, 'questions': questions}]
    values = [f'Enumerable value: {value}' for value in labels.values]
    values += [f'Copy token at context position {i}: {token}' for i, token in enumerate(example.context_tokens)]
    requests.append({
        'state': {**history, 'value_context_tokens': example.context_tokens},
        'questions': {'value': choice(values, 'Assuming an action is being taken next, select the value to fill '
                                     'its input, using `history_tokens` and `value_context_tokens`. '
                                     'Select the literal enumerable value or the observed entity placeholder token. '
                                     'Do not generate a new value.')},
    })
    if task == 'cds' and example.candidates:
        if len(example.candidates) != 100:
            raise ValueError('Reference utterance ranking requires exactly 100 candidates')
        texts = []
        for utterance_id in example.candidates:
            if not 0 <= utterance_id < len(utterances):
                raise ValueError(f'Unknown utterance candidate ID: {utterance_id}')
            if raw:
                text = utterances[utterance_id]
            else:
                if tokenizer is None:
                    raise ValueError('Reference response candidates require a tokenizer')
                if utterance_cache is None:
                    text = ' '.join(tokenizer.tokenize(utterances[utterance_id])[:510])
                else:
                    if utterance_id not in utterance_cache:
                        utterance_cache[utterance_id] = ' '.join(tokenizer.tokenize(utterances[utterance_id])[:510])
                    text = utterance_cache[utterance_id]
            texts.append(text)
        requests.append({
            'state': history,
            'questions': {'utterance': choice(texts, 'Assuming the agent responds with speech next, which candidate '
                                              'is the best immediate continuation of `history_tokens`? '
                                              'Select the most contextually appropriate response, not a later step.')},
        })
    if raw:
        for request in requests:
            request['questions'] = {
                name: Choice(instructions=q.instructions.replace('history_tokens', 'history'), criteria=q.criteria)
                for name, q in request['questions'].items()
            }
    return requests


def serialize_requests(requests: list[JudgmentRequest]) -> list[dict]:
    """Convert SDK question objects to the existing JSONL representation."""
    return [{'state': r['state'], 'questions': {k: q.model_dump(mode='json') for k, q in r['questions'].items()}}
            for r in requests]
