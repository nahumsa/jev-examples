"""Bounded model calls and SDK-response normalization; no benchmark scoring."""

import asyncio
from collections.abc import Mapping
from typing import Protocol

from typesafe_sdk import ChoiceAnswer, JSONContent, Question, SystemOneResponse

from .contracts import HeadJudgment, JudgmentBatch
from .requests import JudgmentRequest


class JudgmentClient(Protocol):
    """Small client interface; tests can supply a fake without credentials."""

    async def system_one(self, state: JSONContent, questions: Mapping[str, Question],
                         *, model: str) -> SystemOneResponse: ...


async def execute_requests(client: JudgmentClient, requests: list[JudgmentRequest],
                           model: str, semaphore: asyncio.Semaphore) -> JudgmentBatch:
    """Execute independent heads and normalize answers into plain Python data."""
    async def ask(request: JudgmentRequest) -> SystemOneResponse:
        async with semaphore:
            return await client.system_one(request['state'], request['questions'], model=model)

    # Drain all heads before raising an error; do not leave sibling calls running.
    responses = await asyncio.gather(*(ask(r) for r in requests), return_exceptions=True)
    heads = {}
    models = set()
    input_tokens = output_tokens = 0
    # Preserve request-order failure reporting, after every call has settled.
    for response in responses:
        if isinstance(response, BaseException):
            raise response
    for request, response in zip(requests, responses):
        if isinstance(response, BaseException):
            raise response  # Explicit narrowing for static type checkers.
        models.add(response.model)
        input_tokens += response.usage.input_tokens or 0
        output_tokens += response.usage.output_tokens or 0
        for name, question in request['questions'].items():
            answer = response.answers[name]
            if not isinstance(answer, ChoiceAnswer):
                raise TypeError(f'Expected Choice answer for {name}')
            heads[name] = HeadJudgment(
                probabilities=tuple(answer.probabilities.get(str(i), 0.0) for i in range(len(question.criteria))),
                confidence=answer.confidence,
            )
    return JudgmentBatch(heads=heads, resolved_models=tuple(sorted(models)),
                         input_tokens=input_tokens, output_tokens=output_tokens,
                         requests=len(requests))
