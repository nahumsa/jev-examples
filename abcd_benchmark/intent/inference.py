"""Pydantic AI execution and normalization of intent prediction responses."""

import json
import time
from dataclasses import asdict

from pydantic_ai import Agent, AgentRunResult
from pydantic_core import to_jsonable_python

from .schema import IntentPrediction


def decode_result(row: dict, result: AgentRunResult[IntentPrediction], *, latency_seconds: float = 0.0) -> dict:
    """Convert an agent result to the established JSON-safe prediction schema."""
    details = result.response.provider_details or {}
    predicted = result.output.intent.value
    probabilities = details.get('probabilities', {}).get('intent', {})
    return to_jsonable_python({
        **row, 'output': result.output.model_dump(mode='json'), 'predicted': predicted,
        'score': details.get('confidence', {}).get('intent'),
        'predicted_probability': probabilities.get(predicted),
        'correct': predicted == row['gold'], 'provider_details': details,
        'resolved_model': result.response.model_name, 'usage': asdict(result.usage),
        'latency_seconds': latency_seconds,
    })


async def predict_row(agent: Agent, row: dict) -> dict:
    """Send only observed state; keep labels exclusively in local scoring."""
    started = time.perf_counter()
    result = await agent.run(json.dumps(row['state']))
    return decode_result(row, result, latency_seconds=time.perf_counter() - started)
