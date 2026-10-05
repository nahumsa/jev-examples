"""Pinned ABCD data downloads shared by setup and evaluation."""

import gzip
import json
from pathlib import Path
from urllib.request import urlopen

DATA_REVISION = '6b8700ce67c6b37b062dd7a60abc76d7ef832a97'
DATA_URL = f'https://raw.githubusercontent.com/asappresearch/abcd/{DATA_REVISION}/data/'
DATA_FILES = (
    'abcd_v1.1.json.gz',
    'abcd_sample.json',
    'ontology.json',
    'guidelines.json',
    'kb.json',
)


def download(filename: str, directory: Path) -> Path:
    """Cache a pinned upstream artifact; don't leave partial files on failure."""
    path = directory / filename
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL + filename, timeout=120) as response:
            content = response.read()
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    return path


def read_json(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)
