"""File provenance and evaluation-artifact lifecycle; no model dependencies."""

import hashlib
import json
from pathlib import Path


def artifact_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ensure_new_artifacts(output: Path) -> Path:
    """Refuse to overwrite either artifact; return the intended summary path."""
    summary_path = output.with_suffix('.summary.json')
    if output.exists() or summary_path.exists():
        raise FileExistsError(f'Output artifacts already exist: {output}')
    return summary_path


def write_summary(path: Path, summary: dict) -> None:
    """Write a runner's summary after its output-artifact guard has succeeded."""
    path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
