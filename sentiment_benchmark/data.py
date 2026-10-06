"""Pinned UCI download and strict, lossless parsing of its three text files."""

import hashlib
import random
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
SHA256 = "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"
SOURCES = ("amazon_cells", "imdb", "yelp")
DEFAULT_DATA = Path("data/sentiment/sentiment-labelled-sentences.zip")
LABELS = ("negative", "positive")
VALIDATION_SPLIT_VERSION = "normalized-text-sha256-v1"
VALIDATION_SPLIT_SEED = 42


@dataclass(frozen=True)
class Sentence:
    id: str
    source: str
    text: str
    gold: str


def download(path: Path) -> None:
    if path.exists():
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read(10_000_001)
    if len(payload) > 10_000_000 or hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("UCI archive checksum mismatch; download not saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)


def parse_sentences(text: str, source: str) -> list[Sentence]:
    rows = []
    # Do NOT use csv's quote handling or str.splitlines(): IMDb contains quotes
    # and Unicode line separators inside sentences, not record boundaries.
    for number, line in enumerate(text.split("\n"), start=1):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        parts = line.rsplit("\t", 1)
        if len(parts) != 2 or not parts[0].strip() or parts[1] not in {"0", "1"}:
            raise ValueError(f"Invalid sentence/label at {source}:{number}")
        rows.append(
            Sentence(f"{source}:{number}", source, parts[0], LABELS[int(parts[1])])
        )
    return rows


def load_dataset(path: Path) -> list[Sentence]:
    if not path.exists():
        raise ValueError("Dataset not found; use --download or provide --data")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError(
            "UCI archive checksum mismatch; remove the cache and download again"
        )
    rows = []
    with ZipFile(path) as archive:
        for source in SOURCES:
            member = f"sentiment labelled sentences/{source}_labelled.txt"
            parsed = parse_sentences(archive.read(member).decode("utf-8"), source)
            if Counter(row.gold for row in parsed) != {
                "negative": 500,
                "positive": 500,
            }:
                raise ValueError(f"Expected 500 sentences per label for {source}")
            rows.extend(parsed)
    return rows


def split_for(sentence: Sentence) -> str:
    """Keep normalized duplicates in one fixed split, including across sources."""
    normalized = " ".join(
        unicodedata.normalize("NFKC", sentence.text).casefold().split()
    )
    key = f"{VALIDATION_SPLIT_VERSION}:{VALIDATION_SPLIT_SEED}:{normalized}"
    bucket = (
        int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % 100
    )
    return "validation" if bucket < 20 else "development"


def validation_sentences(rows: list[Sentence]) -> list[Sentence]:
    return [row for row in rows if split_for(row) == "validation"]


def select_sentences(
    rows: list[Sentence], *, source: str, limit: int, seed: int
) -> list[Sentence]:
    sources = SOURCES if source == "all" else (source,)
    if any(value not in SOURCES for value in sources):
        raise ValueError("Unknown source")
    strata = [(value, label) for value in sources for label in LABELS]
    if limit < 1 or limit % len(strata):
        raise ValueError(f"--limit must be positive and a multiple of {len(strata)}")
    count = limit // len(strata)
    selected = []
    for value, label in strata:
        pool = sorted(
            (r for r in rows if r.source == value and r.gold == label),
            key=lambda r: r.id,
        )
        if count > len(pool):
            raise ValueError(
                f"Not enough validation sentences for {value}/{label}: "
                f"requested {count}, available {len(pool)}. Reduce --limit."
            )
        rng = random.Random(f"sentiment-v1:{seed}:{value}:{label}")
        selected.extend(rng.sample(pool, count))
    random.Random(seed).shuffle(selected)
    return selected
