#!/usr/bin/env python3
"""Prepare frozen original-quality video clips and ASR, independently of QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from beyond_recall_eval.dataset import BeyondRecallDataset
from beyond_recall_eval.sampling import select_trajectories
from beyond_recall_eval.video_segments import APITranscriber, prepare_video, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sample-manifest', type=Path, help='Optional frozen question selection; omit for the full dataset.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--segment-seconds', type=float, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--asr-model', default='FunAudioLLM/SenseVoiceSmall')
    parser.add_argument('--previous-config', type=Path,
                        default=PACKAGE / 'examples/api_config.py')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1–4 preparation workers.')
    if args.previous_config:
        sys.path.insert(0, str(args.previous_config.resolve().parent))
        spec = importlib.util.spec_from_file_location('private_previous_config', args.previous_config)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        os.environ['BR_ASR_API_KEY'] = previous.FULL_CONTEXT_MULTIMODAL_API_KEY
        os.environ['BR_ASR_BASE_URL'] = previous.FULL_CONTEXT_MULTIMODAL_BASE_URL.split('/chat/completions')[0]
    asr_config = {'model': args.asr_model, 'revision': 'unversioned-provider-engineering-test',
                  'api_key_env': 'BR_ASR_API_KEY', 'base_url_env': 'BR_ASR_BASE_URL',
                  'timeout_seconds': 120, 'max_retries': 2}
    trajectories = select_trajectories(list(BeyondRecallDataset(args.dataset).trajectories()),
                                      args.sample_manifest, args.dataset)
    assets = {a.relative_path: a for t in trajectories
              for a in [*(a for s in t.sessions for turn in s.turns for a in turn.media),
                        *(a for q in t.questions for a in q.media)] if a.media_type == 'video'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'manifest.json'
    manifest = {'schema_version': 1, 'purpose': 'question-independent video clips paired with ASR',
                'target_segment_seconds': args.segment_seconds, 'asr': asr_config, 'items': {}}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['target_segment_seconds'] != args.segment_seconds or existing['asr'] != asr_config:
            raise ValueError('Output cache has different settings.')
        manifest['items'].update(existing['items'])

    def work(asset):
        digest = sha(asset.local_path.read_bytes())
        directory = args.output_dir / digest
        record = prepare_video(asset.local_path, directory, segment_seconds=args.segment_seconds,
                               transcriber=APITranscriber(asr_config), asr_config=asr_config)
        return digest, {**record, 'directory': digest}

    errors = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, a): name for name, a in assets.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                digest, item = future.result()
                manifest['items'][digest] = item
                print(json.dumps({'asset': name, 'status': 'ok', 'clips': len(item['segments']),
                                  'frames': item['original_frame_count']}, ensure_ascii=False), flush=True)
                temporary = manifest_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
                temporary.replace(manifest_path)
            except Exception as error:
                errors[name] = str(error)
                print(json.dumps({'asset': name, 'status': 'error', 'error': str(error)}), flush=True)
    report = {'expected_video_assets': len(assets), 'errors': errors, 'successful_assets': len(assets) - len(errors),
              'manifest_sha256': sha(manifest_path.read_bytes()) if manifest_path.exists() else None}
    (args.output_dir / 'preparation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
