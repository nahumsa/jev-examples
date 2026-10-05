"""Provider-independent results passed from inference to prediction decoding."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadJudgment:
    """Probabilities in request-choice order, plus provider confidence."""

    probabilities: tuple[float, ...]
    confidence: float


@dataclass(frozen=True)
class JudgmentBatch:
    """Normalized answers and usage for all independent heads of one example."""

    heads: dict[str, HeadJudgment]
    resolved_models: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    requests: int
