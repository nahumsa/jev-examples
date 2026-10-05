"""Load and prepare benchmark datasets, with no inference or scoring."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abcd_benchmark.common.artifacts import artifact_hash
from abcd_benchmark.common.data import download, read_json
from abcd_benchmark.common.selection import select_conversations

from .config import BenchmarkConfig
from .examples import build_examples
from .schema import (
    PROTOCOL,
    RAW_PROTOCOL,
    TOKENIZER_REVISION,
    Example,
    Labels,
    Tokenizer,
)
from .tokenization import load_tokenizer


@dataclass(frozen=True)
class PreparedBenchmark:
    protocol: str
    labels: Labels
    tokenizer: Tokenizer | None
    utterances: list[str]
    examples: list[Example]
    source_hashes: dict[str, str]
    selected_conversations: int
    total_selected_examples: int
    full_split_evaluation: bool


def load_benchmark_data(config: BenchmarkConfig, *,
                        download_artifact: Callable[[str, Path], Path] = download,
                        load_json: Callable[[Path], Any] = read_json,
                        tokenizer_factory: Callable[[Path, dict], Tokenizer] = load_tokenizer) -> PreparedBenchmark:
    """Keep optional reference-tokenizer loading behind an explicit boundary."""
    raw = config.preprocessing == 'raw'
    ontology_path = download_artifact('ontology.json', config.data_dir)
    ontology = load_json(ontology_path)
    labels = Labels.from_ontology(ontology)
    tokenizer = None if raw else tokenizer_factory(config.data_dir, ontology)
    filename = 'abcd_sample.json' if config.split == 'sample' else 'abcd_v1.1.json.gz'
    dataset = config.dataset or download_artifact(filename, config.data_dir)
    data = load_json(dataset)
    conversations = select_conversations(data, config.split, config.limit, config.seed)
    if not conversations:
        raise ValueError('No conversations selected')
    utterances_path = download_artifact('utterances.json', config.data_dir) if config.task == 'cds' else None
    utterances = load_json(utterances_path) if utterances_path else []
    examples = build_examples(conversations, config.task, labels, tokenizer, raw=raw)
    total_selected_examples = len(examples)
    if config.example_limit is not None:
        examples = examples[:config.example_limit]
    if not examples:
        raise ValueError('No benchmark examples selected')
    source_hashes = {'dataset': artifact_hash(dataset), 'ontology': artifact_hash(ontology_path)}
    if utterances_path:
        source_hashes['utterances'] = artifact_hash(utterances_path)
    if not raw:
        vocab_path = config.data_dir / 'tokenizers' / TOKENIZER_REVISION / 'vocab.txt'
        source_hashes['tokenizer_vocab'] = artifact_hash(vocab_path)
    full_split = data if isinstance(data, list) else data[config.split]
    return PreparedBenchmark(
        protocol=RAW_PROTOCOL if raw else PROTOCOL, labels=labels, tokenizer=tokenizer,
        utterances=utterances, examples=examples, source_hashes=source_hashes,
        selected_conversations=len(conversations), total_selected_examples=total_selected_examples,
        full_split_evaluation=len(conversations) == len(full_split) and len(examples) == total_selected_examples,
    )
