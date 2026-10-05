"""Pure reference copy-context and value-target semantics."""

from .schema import Labels, Tokenizer


def copy_context(history: list[str], action: str, tokenizer: Tokenizer) -> list[str]:
    """Unique tokens >2 chars, retaining the tail as in value_to_id."""
    filtered = []
    for utterance in history:
        # Deliberately match reference split('|')[1] semantics.
        text = utterance.split('|')[1]
        for token in tokenizer.tokenize(text):
            if token not in filtered and len(token) > 2:
                filtered.append(token)
    effective_max = 100 - (len(tokenizer.tokenize(action)) + 3)
    return filtered[-effective_max:]


def value_target(tokens: list[str], action: str, value: str, labels: Labels) -> int:
    """Preserve the reference's first matching category/placeholder rule."""
    for category in labels.value_by_action[action]:
        if category in labels.enumerable:
            if value in labels.enumerable[category]:
                return labels.values.index(value)
        elif f'<{category}>' in tokens:
            return len(labels.values) + tokens.index(f'<{category}>')
    return -1
