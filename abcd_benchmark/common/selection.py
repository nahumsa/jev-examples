"""Reproducible conversation selection shared by both evaluation pipelines."""

import random


def select_conversations[T](data: list[T] | dict[str, list[T]], split: str,
                            limit: int | None, seed: int) -> list[T]:
    """Sample reproducibly, or retain dataset order for a complete split."""
    if isinstance(data, list):
        if split != 'sample':
            raise ValueError('A list dataset is the training sample; use --split sample')
        conversations = data
    else:
        conversations = data[split]
    if limit is None or limit >= len(conversations):
        return list(conversations)
    return random.Random(seed).sample(conversations, limit)
