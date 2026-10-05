"""Jev inference exclusively through PydanticAI; no fallback model or tools."""
import json
import time
from dataclasses import asdict

from pydantic_ai import Agent
from pydantic_core import to_jsonable_python

from .config import AGENT_RETRIES, DEFAULT_MODEL, REQUEST_TIMEOUT_SECONDS, validate_model
from .schema import Prediction


def make_agent(model: str = DEFAULT_MODEL) -> Agent:
    validate_model(model)
    return Agent(model, output_type=Prediction, retries=AGENT_RETRIES,
                 model_settings={'timeout': REQUEST_TIMEOUT_SECONDS})


async def predict(agent: Agent, row: dict) -> dict:
    started = time.perf_counter()
    result = await agent.run(json.dumps(row['state'], ensure_ascii=False))
    predicted = result.output.intent.value
    details = result.response.provider_details or {}
    return to_jsonable_python({
        **row, 'predicted': predicted, 'correct': predicted == row['gold'],
        'confidence': details.get('confidence', {}).get('intent'),
        'probabilities': details.get('probabilities', {}).get('intent', {}),
        'provider_details': details, 'resolved_model': result.response.model_name,
        'usage': asdict(result.usage), 'latency_seconds': time.perf_counter() - started,
    })
