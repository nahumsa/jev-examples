"""Dataset loading and preparation for intent-routing runs."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abcd_benchmark.common.data import download, read_json
from abcd_benchmark.common.selection import select_conversations

from .config import EvaluationConfig
from .requests import prepare_row
from .schema import Intent


@dataclass(frozen=True)
class PreparedEvaluation:
    dataset: Path
    intent_to_flow: dict[str, str]
    rows: list[dict]


def load_evaluation_data(config: EvaluationConfig, *,
                         download_artifact: Callable[[str, Path], Path] = download,
                         load_json: Callable[[Path], Any] = read_json) -> PreparedEvaluation:
    """Own dataset I/O; injectable readers keep orchestration tests offline."""
    ontology = load_json(download_artifact('ontology.json', config.data_dir))
    intent_to_flow = {
        intent: flow for flow, intents in ontology['intents']['subflows'].items()
        for intent in intents
    }
    if set(intent_to_flow) != {intent.value for intent in Intent}:
        raise ValueError('Output enum does not match the upstream intent ontology')
    filename = 'abcd_sample.json' if config.split == 'sample' else 'abcd_v1.1.json.gz'
    dataset = config.dataset or download_artifact(filename, config.data_dir)
    conversations = select_conversations(load_json(dataset), config.split, config.limit, config.seed)
    if not conversations:
        raise ValueError('No conversations selected')
    rows = [prepare_row(c, config, dataset, intent_to_flow) for c in conversations]
    return PreparedEvaluation(dataset=dataset, intent_to_flow=intent_to_flow, rows=rows)
