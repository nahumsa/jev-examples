"""Compose Jev network execution and offline prediction decoding.

Exports the established request helpers alongside the prediction entry point.
"""

import asyncio
import time

from .inference import JudgmentClient, execute_requests
from .predictions import decode_prediction
from .requests import JudgmentRequest, choice, prepare_requests, serialize_requests
from .schema import Example, Labels, Task

__all__ = ['choice', 'predict_example', 'prepare_requests', 'serialize_requests']


async def predict_example(client: JudgmentClient, example: Example,
                          labels: Labels, task: Task, requests: list[JudgmentRequest],
                          model: str, semaphore: asyncio.Semaphore) -> dict:
    """Compose bounded network execution with offline prediction decoding."""
    started = time.perf_counter()
    judgments = await execute_requests(client, requests, model, semaphore)
    row = decode_prediction(example, labels, task, judgments)
    row['latency_seconds'] = time.perf_counter() - started
    return row
