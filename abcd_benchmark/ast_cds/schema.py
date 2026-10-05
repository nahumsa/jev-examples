"""Offline benchmark types, label order, and versioned protocol constants."""

from dataclasses import dataclass
from typing import Literal, Protocol

Task = Literal['ast', 'cds']
TOKENIZER_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
PROTOCOL = 'abcd-v1.1-reference-bert-512'
RAW_PROTOCOL = 'abcd-v1.1-raw-full-history-entities'
MULTI_SLOT = {'verify-identity', 'validate-purchase'}


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class Example:
    example_id: str
    convo_id: int
    turn_count: int
    slot_index: int
    terminal: bool
    history_tokens: list[str]
    context_tokens: list[str]
    candidates: list[int]
    gold: dict[str, int]
    history_text: str | None = None

    def metadata(self) -> dict:
        return {
            'example_id': self.example_id, 'convo_id': self.convo_id,
            'turn_count': self.turn_count, 'slot_index': self.slot_index,
            'terminal': self.terminal, 'gold': self.gold,
        }


@dataclass(frozen=True)
class Labels:
    intents: list[str]
    actions: list[str]
    values: list[str]
    nextsteps: list[str]
    value_by_action: dict[str, list[str]]
    enumerable: dict[str, list[str]]

    @classmethod
    def from_ontology(cls, ontology: dict) -> 'Labels':
        # Preserve upstream order and its pre-lowercase duplicate check exactly.
        values = []
        for options in ontology['values']['enumerable'].values():
            for value in options:
                if value not in values:
                    values.append(value.lower())
        return cls(
            intents=[v for options in ontology['intents']['subflows'].values() for v in options],
            actions=[v for buttons in ontology['actions'].values() for v in buttons],
            values=values, nextsteps=ontology['next_steps'],
            value_by_action={a: slots for buttons in ontology['actions'].values() for a, slots in buttons.items()},
            enumerable={k: [v.lower() for v in vs] for k, vs in ontology['values']['enumerable'].items()},
        )
