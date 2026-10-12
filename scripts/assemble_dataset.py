#!/usr/bin/env python3
"""Combine public text and separately downloaded media, verifying source hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from beyond_recall_eval.manifest import dataset_fingerprint


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Invalid path in source checksum list.')
    return path


def assemble(text_root, media_root, output, mode='copy'):
    text_root, media_root, output = map(lambda p: Path(p).resolve(),
                                        (text_root, media_root, output))
    if output.exists():
        raise FileExistsError('Output already exists; choose a new dataset directory.')
    metadata = json.loads((text_root / 'TEXT_RELEASE.json').read_text())
    checksum_file = text_root / 'SOURCE_SHA256SUMS'
    entries = [line.split('  ', 1) for line in checksum_file.read_text().splitlines()]
    if len({name for _, name in entries}) != len(entries):
        raise ValueError('Duplicate paths in source checksum list.')
    for _, name in entries:
        safe_relative(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.beyond-recall-assembly-',
                                     dir=output.parent) as temporary:
        staging = Path(temporary)
        for expected, name in entries:
            relative = safe_relative(name)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in metadata['task_files']:
                source = text_root / relative
                if sha(source) != metadata['task_files'][name]['text_sha256']:
                    raise ValueError('Public text checksum mismatch: ' + name)
                content = source.read_text()
                for escaped, codepoint in metadata.get('json_byte_restorations', {}).get(name, {}).items():
                    content = content.replace(escaped, chr(codepoint))
                target.write_text(content)
                if sha(target) != metadata['task_files'][name]['source_sha256']:
                    raise ValueError('Source JSON restoration failed: ' + name)
            else:
                source = text_root / ('SOURCE_README.md' if name == 'README.md' else relative)
                if not source.is_file():
                    source = media_root / relative
                if not source.is_file():
                    raise FileNotFoundError('Missing media-release file: ' + name)
                if sha(source) != expected:
                    raise ValueError('Media-release checksum mismatch: ' + name)
                if mode == 'hardlink':
                    os.link(source.resolve(), target)
                else:
                    shutil.copyfile(source, target)
            if sha(target) != expected:
                raise ValueError('Assembled file checksum mismatch: ' + name)
        shutil.copyfile(checksum_file, staging / 'SHA256SUMS')
        actual = dataset_fingerprint(staging)
        if actual != metadata['full_dataset_fingerprint']:
            raise ValueError('Assembled dataset differs from the frozen release.')
        staging.rename(output)
    return {'output': str(output), 'verified_files': len(entries),
            'dataset_fingerprint': actual, 'mode': mode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-root', type=Path,
                        default=PACKAGE / 'data/BeyondRecall_text')
    parser.add_argument('--media-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['copy', 'hardlink'], default='copy',
                        help='Hardlinks require the same filesystem; do not edit linked input files.')
    args = parser.parse_args()
    print(json.dumps(assemble(args.text_root, args.media_root, args.output, args.mode)))


if __name__ == '__main__':
    main()
