"""Download ABCD locally from a pinned upstream revision (no API key needed)."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from abcd_benchmark.common.data import (
    DATA_FILES,
    DATA_REVISION,
    DATA_URL,
    download,
    read_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/abcd'))
    parser.add_argument('--unpack', action='store_true', help='Also create abcd_v1.1.json')
    parser.add_argument('--include-utterances', action='store_true',
                        help='Also download the utterance-ranking lookup table')
    args = parser.parse_args()
    files = list(DATA_FILES)
    if args.include_utterances:
        files.append('utterances.json')
    manifest = {'revision': DATA_REVISION, 'source': DATA_URL, 'files': {}}
    for filename in files:
        path = download(filename, args.data_dir)
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['files'][filename] = {
            'bytes': path.stat().st_size,
            'sha256': checksum,
        }
        print(f'{path} ({path.stat().st_size:,} bytes)')

    compressed = args.data_dir / 'abcd_v1.1.json.gz'
    if args.unpack:
        unpacked = compressed.with_suffix('')
        if not unpacked.exists():
            temporary = unpacked.with_suffix('.json.tmp')
            try:
                with gzip.open(compressed, 'rb') as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(unpacked)
            finally:
                temporary.unlink(missing_ok=True)
        print(unpacked)

    data = read_json(compressed)
    manifest['split_counts'] = {split: len(rows) for split, rows in data.items()}
    (args.data_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    print(f"Conversations: {manifest['split_counts']}")


if __name__ == '__main__':
    main()
