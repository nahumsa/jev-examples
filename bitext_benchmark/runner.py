"""Sequential bounded execution with exclusive, flushed artifacts."""
import json
from pathlib import Path

from .config import SUMMARY_SUFFIX, validate_output
from .metrics import score
from .model import make_agent, predict
from .schema import Prediction


async def run(rows: list[dict], *, output: Path, manifest: dict, model: str,
              dry_run: bool, agent=None) -> dict:
    validate_output(output)
    summary_path = output.with_suffix(SUMMARY_SUFFIX)
    output.parent.mkdir(parents=True, exist_ok=True)
    # No agent is created in dry-run mode (no API key required).
    if not dry_run and agent is None:
        agent = make_agent(model)
    completed = []
    status = 'running'
    with output.open('x', encoding='utf-8') as predictions, summary_path.open('x', encoding='utf-8') as summary_file:
        try:
            for row in rows:
                record = dict(row) if dry_run else await predict(agent, row)
                predictions.write(json.dumps(record, ensure_ascii=False) + '\n')
                predictions.flush()
                completed.append(record)
            status = 'complete'
        finally:
            summary = {'status': status if status == 'complete' else 'partial',
                       'dry_run': dry_run, 'model': model, 'manifest': manifest,
                       'selected_ids': [r['id'] for r in rows],
                       'selected_count': len(rows), 'completed_count': len(completed),
                       'output_schema': Prediction.model_json_schema(),
                       'metrics': None if dry_run else score(completed)}
            summary_file.write(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary
