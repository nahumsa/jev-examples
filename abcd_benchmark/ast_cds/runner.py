"""Durable AST/CDS orchestration with bounded API concurrency."""

import asyncio
import json
import time
from contextlib import AsyncExitStack

from typesafe_sdk import AsyncTypeSafeClient

from abcd_benchmark.common.artifacts import (
    artifact_hash,
    ensure_new_artifacts,
    write_summary,
)
from abcd_benchmark.common.data import DATA_REVISION, download, read_json

from .config import BenchmarkConfig
from .datasets import load_benchmark_data
from .pipeline import predict_example, prepare_requests, serialize_requests
from .reporting import build_summary
from .tokenization import load_tokenizer

__all__ = ['BenchmarkConfig', 'artifact_hash', 'run_benchmark']


async def run_benchmark(config: BenchmarkConfig) -> dict:
    config.validate()
    summary_path = ensure_new_artifacts(config.output)
    prepared = load_benchmark_data(config, download_artifact=download, load_json=read_json,
                                   tokenizer_factory=load_tokenizer)
    examples, labels = prepared.examples, prepared.labels
    config.output.parent.mkdir(parents=True, exist_ok=True)
    rows, failed_examples = [], []
    cache: dict[int, str] = {}
    prepared_request_count = 0
    started = time.perf_counter()
    complete = False
    semaphore = asyncio.Semaphore(config.concurrency)
    async with AsyncExitStack() as stack:
        client = None if config.dry_run else await stack.enter_async_context(AsyncTypeSafeClient(timeout=60))
        with config.output.open('x', encoding='utf-8') as stream:
            try:
                for start in range(0, len(examples), config.concurrency):
                    batch = examples[start:start + config.concurrency]
                    requests = [prepare_requests(e, labels, config.task, prepared.utterances, prepared.tokenizer, cache)
                                for e in batch]
                    if client is None:
                        results = [e.metadata() for e in batch]
                    else:
                        results = await asyncio.gather(*(
                            predict_example(client, e, labels, config.task, reqs, config.model, semaphore)
                            for e, reqs in zip(batch, requests)
                        ), return_exceptions=True)
                    errors = []
                    for example, reqs, result in zip(batch, requests, results):
                        if isinstance(result, BaseException):
                            failed_examples.append({'example_id': example.example_id, 'error_type': type(result).__name__})
                            errors.append(result)
                            continue
                        result.update(
                            task=config.task, protocol=prepared.protocol, data_revision=DATA_REVISION,
                            split=config.split, dry_run=config.dry_run,
                            requested_model=config.model, requests=serialize_requests(reqs),
                        )
                        stream.write(json.dumps(result) + '\n')
                        stream.flush()
                        rows.append(result)
                        prepared_request_count += len(reqs)
                    if errors:
                        raise errors[0]
                    if len(rows) % 100 < config.concurrency or len(rows) == len(examples):
                        print(f'{config.task.upper()}: saved {len(rows)}/{len(examples)} examples', flush=True)
                complete = True
            finally:
                summary = build_summary(config, prepared, rows, complete=complete,
                                        failed_examples=failed_examples, prepared_request_count=prepared_request_count,
                                        wall_time_seconds=time.perf_counter() - started)
                write_summary(summary_path, summary)
    return summary
