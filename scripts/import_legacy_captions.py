#!/usr/bin/env python3
"""Map legacy visual captions to release media by exact source-content SHA."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from beyond_recall_eval.dataset import BeyondRecallDataset
from beyond_recall_eval.rendering import caption_prompt_fingerprint
from beyond_recall_eval.manifest import hash_json, hash_file
from beyond_recall_eval.sampling import select_trajectories
from beyond_recall_eval.video_segments import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sample-manifest', type=Path, help='Optional frozen question selection; omit for the full dataset.')
    parser.add_argument('--legacy-dataset', type=Path, required=True)
    parser.add_argument('--legacy-cache', type=Path, action='append', default=[])
    parser.add_argument('--legacy-cache-dir', type=Path, help='Also inspect all *captions*.json caches in this directory.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=PACKAGE / 'config/api_pdf_cropped_10.json')
    parser.add_argument('--previous-config', type=Path, default=PACKAGE / 'examples/api_config.py')
    parser.add_argument('--legacy-source-root', type=Path, help='Root for relative paths in historical captions.')
    parser.add_argument('--mmdu-mapping', type=Path, help='Optional historical MMDU URL mapping.')
    parser.add_argument('--locomo-mapping', type=Path, help='Optional historical LoCoMo URL mapping.')
    args = parser.parse_args()
    source_root = (args.legacy_source_root or args.legacy_dataset).resolve()
    if args.legacy_cache_dir:
        args.legacy_cache.extend(sorted(args.legacy_cache_dir.glob('*captions*.json')))
    sys.path.insert(0, str(args.previous_config.resolve().parent))
    spec = importlib.util.spec_from_file_location('private_previous_config', args.previous_config)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    config = json.loads(args.config.read_text())
    source_aliases = {}
    mmdu_mapping = args.mmdu_mapping
    if mmdu_mapping is not None:
        base = getattr(previous, 'MMDU_IMAGE_PUBLIC_BASE_URL', '').rstrip('/')
        for row in json.loads(mmdu_mapping.read_text()).get('records', []):
            if base and row.get('source_path') and row.get('upload_filename'):
                source_aliases[base + '/' + row['upload_filename']] = row['source_path']
    locomo_mapping = args.locomo_mapping
    if locomo_mapping is not None:
        base = getattr(previous, 'LOCOMO_IMAGE_PUBLIC_BASE_URL', '').rstrip('/')
        for row in json.loads(locomo_mapping.read_text()).get('records', []):
            if row.get('filename'):
                local = str(locomo_mapping.parent / row['filename'])
                source_aliases[row.get('url', '')] = local
                if base:
                    source_aliases[base + '/' + row['filename']] = local
    trajectories = select_trajectories(list(BeyondRecallDataset(args.dataset).trajectories()), args.sample_manifest, args.dataset)
    assets = {a.relative_path: a for t in trajectories for a in
              [*(a for s in t.sessions for turn in s.turns for a in turn.media), *(a for q in t.questions for a in q.media)]}
    wanted_sizes = {a.local_path.stat().st_size for a in assets.values()}
    by_hash, path_hashes = {}, {}

    def register(source, caption, origin, priority):
        if not isinstance(source, str) or not isinstance(caption, str) or not caption.strip():
            return
        source = source_aliases.get(source, source)
        path = Path(source)
        if not path.is_absolute():
            path = source_root / path
        if not path.is_file():
            return
        if path.stat().st_size not in wanted_sizes:
            return
        if path not in path_hashes:
            path_hashes[path] = sha(path.read_bytes())
        digest = path_hashes[path]
        if digest not in by_hash or priority > by_hash[digest]['priority']:
            by_hash[digest] = {'caption': caption.strip(), 'legacy_source': source,
                               'legacy_origin': origin, 'priority': priority}

    def walk(value):
        if isinstance(value, list):
            for row in value:
                walk(row)
        elif isinstance(value, dict):
            video_caption = value.get('video_caption') or value.get('original_video_caption')
            if video_caption:
                register(value.get('original_video_path') or value.get('video_path'), video_caption,
                         'existing video_caption; legacy visual-only preprocessing', 2)
            caption = (value.get('original_image_description') or value.get('image_description')
                       or value.get('original_blip_caption') or value.get('blip_caption'))
            for field in ('image_path', 'image_paths', 'image', 'images'):
                paths = value.get(field, [])
                for path in paths if isinstance(paths, list) else [paths]:
                    register(path, caption, 'existing dataset image description', 1)
            for row in value.values():
                if isinstance(row, (dict, list)):
                    walk(row)

    for legacy_root in [args.legacy_dataset]:
        for path in sorted(legacy_root.glob('*/*.json')):
            walk(json.loads(path.read_text()))
    for cache in args.legacy_cache:
        payload = json.loads(cache.read_text())
        for key, item in payload.get('items', payload).items():
            if isinstance(item, dict):
                metadata = item.get('metadata') or {}
                source = metadata.get('source') or metadata.get('image_path') or key
                register(source, item.get('caption'), 'legacy frozen caption cache', 3)
            elif isinstance(item, str):
                register(key, item, 'legacy frozen caption cache', 3)
    items, missing = {}, []
    for name, asset in assets.items():
        digest = sha(asset.local_path.read_bytes())
        if digest in by_hash:
            items[name] = {'status': 'ok', 'source_sha256': digest, **by_hash[digest]}
        else:
            missing.append(name)
    metadata = {'schema_version': 1, 'generator_model': 'legacy-mixed-captions',
                'generator_revision': 'legacy-unversioned-provider',
                'prompt_fingerprint': hash_json({'mode': 'legacy import', 'caches': [hash_file(p) for p in args.legacy_cache]}),
                'complete': not missing, 'representation': 'legacy_frozen_captions',
                'historical_configured_vision_model': previous.VISION_MODEL,
                'provenance': 'legacy mixed sources, not regenerated with the current prompt',
                'prompt_fingerprint_scope': 'import provenance, not historical generation certification',
                'audio_policy': 'legacy visual captions only; ASR not added',
                'original_models_and_prompts_pinned': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'metadata': metadata, 'items': items}, ensure_ascii=False, indent=2) + '\n')
    report = {'requested_assets': len(assets), 'matched_assets': len(items), 'missing_assets': missing,
              'policy': 'exact source SHA only; changed PDFs/images never assigned unrelated captions'}
    args.output.with_suffix('.coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'requested_assets': len(assets), 'matched_assets': len(items), 'missing_assets': len(missing)}))
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
