"""Bounded intent orchestration and durable output, with injectable I/O."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from abcd_benchmark.common.artifacts import ensure_new_artifacts, write_summary
from abcd_benchmark.common.data import download, read_json

from .agents import make_agent
from .config import EvaluationConfig
from .datasets import load_evaluation_data
from .inference import predict_row
from .reporting import build_summary


async def run_evaluation(config: EvaluationConfig, *,
                         download_artifact: Callable[[str, Path], Path] = download,
                         load_json: Callable[[Path], Any] = read_json,
                         agent_factory: Callable[[str], Agent] = make_agent,
                         predictor: Callable[[Agent, dict], Awaitable[dict]] = predict_row) -> dict:
    """Flush successful calls and summarize partial runs even after failure."""
    config.validate()
    summary_path = ensure_new_artifacts(config.output)
    prepared = load_evaluation_data(config, download_artifact=download_artifact, load_json=load_json)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    complete = False
    agent = None if config.dry_run else agent_factory(config.model)
    with config.output.open('x', encoding='utf-8') as stream:
        try:
            for start in range(0, len(prepared.rows), config.concurrency):
                batch = prepared.rows[start:start + config.concurrency]
                results = batch if agent is None else await asyncio.gather(
                    *(predictor(agent, row) for row in batch), return_exceptions=True,
                )
                errors = []
                for result in results:
                    if isinstance(result, BaseException):
                        errors.append(result)
                        continue
                    stream.write(json.dumps(result) + '\n')
                    stream.flush()
                    rows.append(result)
                if errors:
                    raise errors[0]
                if len(rows) % 100 < config.concurrency or len(rows) == len(prepared.rows):
                    print(f'Completed {len(rows)}/{len(prepared.rows)} conversations', flush=True)
            complete = True
        finally:
            summary = build_summary(config, prepared, rows, complete=complete,
                                    wall_time_seconds=time.perf_counter() - started)
            write_summary(summary_path, summary)
    return summary
