"""Pinned public CSV loading and normalized-message-grouped evaluation splits."""
import csv
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from .config import (
    CSV_COLUMNS, DATASET, DEFAULT_PATH, DEV_BUCKETS, REQUEST_TIMEOUT_SECONDS,
    REVISION, SPLIT_BUCKETS, SPLITS, URL, Split,
)
from .schema import Intent


def normalize(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def load_data(path: Path = DEFAULT_PATH, *, download: bool = False) -> tuple[list[dict], dict]:
    if not path.exists():
        if not download:
            raise FileNotFoundError(f'{path}: use --download or provide --data')
        if path != DEFAULT_PATH:
            raise ValueError('Automatic downloads require the default pinned cache path')
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            content = response.read()
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        required = CSV_COLUMNS
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must contain {sorted(required)}')
        rows = []
        groups: dict[str, str] = {}
        for index, raw in enumerate(reader):
            if any(raw.get(key) is None for key in required):
                raise ValueError(f'Malformed CSV row {index}')
            instruction = raw['instruction']
            if not normalize(instruction):
                raise ValueError(f'Empty instruction at row {index}')
            label = Intent(raw['intent']).value
            normalized = normalize(instruction)
            if normalized in groups and groups[normalized] != label:
                raise ValueError(f'Conflicting labels for normalized instruction at row {index}')
            groups[normalized] = label
            rows.append({'id': f'row-{index:05d}', 'state': {'instruction': instruction},
                         'gold': label, 'category': raw['category'], 'flags': raw['flags']})
    if not rows:
        raise ValueError('Dataset contains no rows')
    manifest = {'dataset': DATASET, 'reference_revision': REVISION,
                'source_url': URL if path == DEFAULT_PATH else None,
                'source_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'rows': len(rows), 'unique_normalized_messages': len(groups),
                'duplicate_rows': len(rows) - len(groups),
                'intent_counts': dict(sorted(Counter(r['gold'] for r in rows).items())),
                'category_counts': dict(sorted(Counter(r['category'] for r in rows).items()))}
    return rows, manifest


def split_for(row: dict, seed: int) -> Split:
    digest = hashlib.sha256(f"{seed}\0{normalize(row['state']['instruction'])}".encode()).digest()
    return 'dev' if int.from_bytes(digest[:8], 'big') % SPLIT_BUCKETS < DEV_BUCKETS else 'test'


def select_rows(rows: list[dict], *, split: Split, seed: int, limit: int | None) -> list[dict]:
    if split not in SPLITS:
        raise ValueError('Split must be dev or test')
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    selected = [row for row in rows if split_for(row, seed) == split]
    # Stable hash ordering avoids selecting contiguous template/intent blocks.
    selected.sort(key=lambda row: hashlib.sha256(f"{seed}\0{row['id']}".encode()).digest())
    return selected if limit is None else selected[:limit]
