"""Dataset preparation and orchestration driven by validated configuration."""
from .config import BenchmarkConfig, SPLITS, SPLIT_VERSION
from .data import load_data, select_rows, split_for
from .runner import run


async def run_evaluation(config: BenchmarkConfig) -> dict:
    """Prepare reproducible requests and execute a dry run or approved inference."""
    rows, manifest = load_data(config.data, download=config.download)
    selected = select_rows(rows, split=config.split, seed=config.seed, limit=config.effective_limit)
    if not selected:
        raise ValueError('Selected split contains no examples')
    manifest.update({
        'split_version': SPLIT_VERSION, 'split': config.split, 'seed': config.seed,
        'split_ids': {split: [row['id'] for row in rows if split_for(row, config.seed) == split]
                      for split in SPLITS},
    })
    print(f'{len(selected)} examples; maximum {0 if config.dry_run else len(selected)} agent calls '
          '(SDK network retries may add requests).')
    return await run(selected, output=config.output, manifest=manifest,
                     model=config.model, dry_run=config.dry_run)
