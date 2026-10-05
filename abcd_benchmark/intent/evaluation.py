"""Compatibility facade for modular intent-routing evaluation.

Explicitly forward legacy module-level dependencies so existing callers and
mock-based tests can retain their original injection points.
"""

from abcd_benchmark.common.data import download, read_json
from abcd_benchmark.common.selection import select_conversations
from abcd_benchmark.intent.agents import make_agent
from abcd_benchmark.intent.config import EvaluationConfig
from abcd_benchmark.intent.inference import predict_row
from abcd_benchmark.intent.requests import prepare_row
from abcd_benchmark.intent.runner import run_evaluation as _run_evaluation

__all__ = ['EvaluationConfig', 'predict_row', 'prepare_row', 'run_evaluation', 'select_conversations']


async def run_evaluation(config: EvaluationConfig) -> dict:
    return await _run_evaluation(config, download_artifact=download, load_json=read_json,
                                 agent_factory=make_agent, predictor=predict_row)
