"""Optional reference-tokenizer I/O, isolated from raw examples and scoring."""

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

from .schema import TOKENIZER_REVISION

if TYPE_CHECKING:
    from transformers import BertTokenizer


def load_tokenizer(data_dir: Path, ontology: dict) -> 'BertTokenizer':
    """Only download a pinned vocabulary, not model weights or remote code."""
    from transformers import BertTokenizer

    path = data_dir / 'tokenizers' / TOKENIZER_REVISION / 'vocab.txt'
    if not path.exists():
        url = f'https://huggingface.co/google-bert/bert-base-uncased/resolve/{TOKENIZER_REVISION}/vocab.txt'
        with urlopen(url, timeout=120) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    tokenizer = BertTokenizer(vocab_file=str(path), do_lower_case=True)
    tokenizer.add_tokens([
        f'<{slot}>' for slots in ontology['values']['non_enumerable'].values() for slot in slots
    ])
    return tokenizer
