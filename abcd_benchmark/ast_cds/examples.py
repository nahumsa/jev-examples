"""Construct reference-compatible AST/CDS decisions from recorded histories.

No gold labels are included in model state. Reference copy-context lengths do
still depend on annotated actions; that benchmark artifact is retained.
Previously public types/helpers remain exported from this module.
"""

import re

from .preprocessing import copy_context, value_target
from .schema import (
    MULTI_SLOT,
    PROTOCOL,
    RAW_PROTOCOL,
    TOKENIZER_REVISION,
    Example,
    Labels,
    Task,
    Tokenizer,
)
from .tokenization import load_tokenizer

__all__ = [
    'MULTI_SLOT', 'PROTOCOL', 'RAW_PROTOCOL', 'TOKENIZER_REVISION',
    'Example', 'Labels', 'Task', 'Tokenizer', 'build_examples',
    'copy_context', 'load_tokenizer', 'value_target',
]


def build_examples(conversations: list[dict], task: Task, labels: Labels,
                   tokenizer: Tokenizer | None, *, raw: bool = False) -> list[Example]:
    """Predict BEFORE each observed agent/action turn; append a terminal CDS row.

    AST drops value-bearing examples whose reference value target cannot be
    found. CDS keeps them. Multiple slots share turn_count and retain their
    original expansion order, including the upstream CDS position omission.
    """
    if not raw and tokenizer is None:
        raise ValueError('Reference preprocessing requires a tokenizer')
    examples = []

    def append(history: list[str], convo_id: int, turn: dict, slot: int,
               value: str, action_context: str, terminal: bool = False) -> None:
        intent, nextstep, action, _, utterance_id = turn['targets']
        if terminal:
            nextstep, utterance_id = 'end_conversation', -1
        is_action = nextstep == 'take_action' and not terminal
        text = '\n'.join(u.split('|', 1)[1] for u in history)
        if raw:
            # Entity choices come only from observed history, independently of gold action.
            context = list(dict.fromkeys(re.findall(r'<[^<>\s]+>', text)))
        else:
            context = copy_context(history, action_context, tokenizer) if is_action and value != 'not applicable' else []
        value_id = value_target(context, action, value, labels) if is_action and value != 'not applicable' else -1
        if task == 'ast' and value != 'not applicable' and value_id < 0:
            return
        # The reference truncates the beginning, NOT the most recent tail.
        history_tokens = [] if raw else tokenizer.tokenize(' [SEP] '.join(u.split('|')[1] for u in history) or '[PAD]')[:510]
        examples.append(Example(
            example_id=f'{convo_id}:{turn["turn_count"]}:{slot}:{int(terminal)}',
            convo_id=convo_id, turn_count=turn['turn_count'], slot_index=slot,
            terminal=terminal, history_tokens=history_tokens, context_tokens=context,
            history_text=text if raw else None,
            candidates=list(turn['candidates']) if nextstep == 'retrieve_utterance' else [],
            gold={'intent': labels.intents.index(intent),
                  'nextstep': labels.nextsteps.index(nextstep),
                  'action': labels.actions.index(action) if is_action else -1,
                  'value': value_id, 'utterance': utterance_id},
        ))

    for conversation in conversations:
        history: list[str] = []
        convo_id = conversation['convo_id']
        for turn in conversation['delexed']:
            speaker = turn['speaker']
            if speaker == 'agent' and task == 'cds':
                append(history, convo_id, turn, 0, 'not applicable', '')
            elif speaker == 'action':
                action, values = turn['targets'][2:4]
                if labels.value_by_action[action]:
                    slots = list(enumerate(values[:3])) if action in MULTI_SLOT else [(0, values[0])]
                    for slot, value in slots:
                        action_context = f'{action} {"abc"[slot]}' if task == 'ast' and action in MULTI_SLOT else action
                        append(history, convo_id, turn, slot, value, action_context)
                else:
                    append(history, convo_id, turn, 0, 'not applicable', action)
            # AST history records action names; CDS history records action text.
            observed = turn['targets'][2] if speaker == 'action' and task == 'ast' else turn['text']
            history.append(f'{speaker}|{observed}')
        if task == 'cds' and conversation['delexed']:
            append(history, convo_id, conversation['delexed'][-1], 0, 'not applicable', '', terminal=True)
    return examples
