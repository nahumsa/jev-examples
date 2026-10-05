import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from typesafe_sdk import ChoiceAnswer

from abcd_benchmark.ast_cds.contracts import HeadJudgment, JudgmentBatch
from abcd_benchmark.ast_cds.examples import Example, Labels
from abcd_benchmark.ast_cds.inference import execute_requests
from abcd_benchmark.ast_cds.metrics import top_k
from abcd_benchmark.ast_cds.predictions import decode_prediction
from abcd_benchmark.ast_cds.requests import choice


@pytest.fixture
def labels():
    return Labels(intents=['refund', 'exchange'], actions=['verify', 'refund'],
                  values=['yes', 'no'], nextsteps=['retrieve_utterance', 'take_action', 'end_conversation'],
                  value_by_action={'verify': ['account_id'], 'refund': []}, enumerable={})


@pytest.fixture
def example():
    return Example(example_id='1:1:0:0', convo_id=1, turn_count=1, slot_index=0, terminal=False,
                   history_tokens=['refund', '<account_id>'], context_tokens=['<account_id>'], candidates=[],
                   gold={'intent': 0, 'nextstep': 1, 'action': 0, 'value': 2, 'utterance': -1})


def judgments(**heads):
    return JudgmentBatch(heads=heads, resolved_models=('fake',), input_tokens=12, output_tokens=0, requests=2)


def test_decode_prediction_is_offline_and_decodes_copy_value(example, labels):
    batch = judgments(intent=HeadJudgment((0.5, 0.5), 0.5),
                      nextstep=HeadJudgment((0.0, 1.0, 0.0), 1.0),
                      action=HeadJudgment((1.0, 0.0), 1.0),
                      value=HeadJudgment((0.0, 0.0, 1.0), 1.0))
    row = decode_prediction(example, labels, 'cds', batch)
    assert row['output']['intent'] == 'refund'  # Preserve argmax's first-index tie rule.
    assert row['output']['value'] == '<account_id>'
    assert row['correctness']['turn'] is True
    assert row['usage'] == {'input_tokens': 12, 'output_tokens': 0, 'requests': 2}
    assert row['latency_seconds'] == 0
    assert row['probabilities']['value'] == [0.0, 0.0, 1.0]


def test_decode_prediction_preserves_reference_retrieval_ties(example, labels):
    example = replace(example, candidates=list(range(100)))
    batch = judgments(intent=HeadJudgment((1.0, 0.0), 1.0),
                      nextstep=HeadJudgment((1.0, 0.0, 0.0), 1.0),
                      action=HeadJudgment((1.0, 0.0), 1.0),
                      value=HeadJudgment((1.0, 0.0, 0.0), 1.0),
                      utterance=HeadJudgment((0.0,) * 100, 0.0))
    row = decode_prediction(example, labels, 'cds', batch)
    assert row['output_ids']['utterance'] == top_k([0.0] * 100, 1)[0]


def test_execute_requests_bounds_concurrency_and_normalizes_answers():
    active = maximum = 0

    class Client:
        async def system_one(self, state, questions, *, model):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1
            return SimpleNamespace(model=model, usage=SimpleNamespace(input_tokens=3, output_tokens=None),
                                   answers={name: ChoiceAnswer(choice='0', confidence=0.8, probabilities={'0': 1.0})
                                            for name in questions})

    requests = [{'state': {}, 'questions': {name: choice(['one', 'two'], 'Choose')}}
                for name in ['action', 'value']]
    batch = asyncio.run(execute_requests(Client(), requests, 'fake', asyncio.Semaphore(1)))
    assert maximum == 1
    assert batch.heads['action'].probabilities == (1.0, 0.0)
    assert batch.input_tokens == 6
    assert batch.output_tokens == 0
    assert batch.requests == 2


def test_execute_requests_drains_sibling_calls_before_raising():
    finished = []

    class Client:
        async def system_one(self, state, questions, *, model):
            name = next(iter(questions))
            if name == 'action':
                raise RuntimeError('offline failure')
            await asyncio.sleep(0)
            finished.append(name)
            return SimpleNamespace(model=model, usage=SimpleNamespace(input_tokens=0, output_tokens=0),
                                   answers={name: ChoiceAnswer(choice='0', confidence=1.0, probabilities={'0': 1.0})})

    requests = [{'state': {}, 'questions': {name: choice(['one', 'two'], 'Choose')}}
                for name in ['action', 'value']]
    with pytest.raises(RuntimeError, match='offline failure'):
        asyncio.run(execute_requests(Client(), requests, 'fake', asyncio.Semaphore(2)))
    assert finished == ['value']
