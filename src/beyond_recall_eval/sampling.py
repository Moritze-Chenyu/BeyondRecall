"""Frozen question selection, stratified across tasks and modalities."""
from __future__ import annotations
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from .manifest import dataset_fingerprint
from .temporal import visible_history


def select_trajectories(trajectories, manifest_path, dataset_root):
    if manifest_path is None:
        return trajectories
    payload = json.loads(Path(manifest_path).read_text())
    if payload['dataset_fingerprint'] != dataset_fingerprint(dataset_root):
        raise ValueError('Sample manifest refers to a different dataset version.')
    keys = payload['question_keys']
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate sample keys.')
    expected = {f'{t.key}/q{q.index}' for t in trajectories for q in t.questions}
    if set(keys) - expected:
        raise ValueError('Sample contains unknown questions.')
    selected = set(keys)
    return [replace(t, questions=tuple(q for q in t.questions if f'{t.key}/q{q.index}' in selected))
            for t in trajectories if any(f'{t.key}/q{q.index}' in selected for q in t.questions)]


def build_sample(trajectories, dataset_root, count=100, seed=42):
    rows = []
    for t in trajectories:
        for q in t.questions:
            history = visible_history(t.sessions, q)
            assets = [a for s in history for turn in s.turns for a in turn.media] + list(q.media)
            types = {a.media_type for a in assets}
            rows.append({'question_key': f'{t.key}/q{q.index}', 'task': t.task,
                         'timestamped': q.timestamp is not None, 'types': sorted(types),
                         'visible_sessions': len(history)})
    if count <= 0 or count > len(rows):
        raise ValueError('Sample count outside dataset size.')
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = {}
    # All corrected time-sensitive questions are included when budget permits.
    temporal = [r for r in rows if r['timestamped']]
    if count >= len(temporal):
        selected.update((r['question_key'], r) for r in temporal)
    # Guarantee task and modality coverage before filling round-robin by task.
    for task in sorted({r['task'] for r in rows}):
        if len(selected) >= count:
            break
        for r in rows:
            if r['task'] == task and r['question_key'] not in selected:
                selected[r['question_key']] = r
                break
    for modality in ('document', 'video', 'image', 'spreadsheet'):
        candidates = [r for r in rows if modality in r['types'] and r['question_key'] not in selected]
        for r in candidates[:min(5, count - len(selected))]:
            selected[r['question_key']] = r
    tasks = sorted({r['task'] for r in rows})
    while len(selected) < count:
        for task in tasks:
            candidate = next((r for r in rows if r['task'] == task and r['question_key'] not in selected), None)
            if candidate:
                selected[candidate['question_key']] = candidate
            if len(selected) == count:
                break
    items = sorted(selected.values(), key=lambda r: r['question_key'])
    return {'schema_version': 1, 'purpose': 'fixed task- and modality-stratified evaluation subset',
            'seed': seed, 'dataset_fingerprint': dataset_fingerprint(dataset_root),
            'question_keys': [r['question_key'] for r in items],
            'task_counts': dict(Counter(r['task'] for r in items)),
            'modality_counts': dict(Counter(m for r in items for m in r['types'])),
            'timestamped_questions': sum(r['timestamped'] for r in items), 'items': items}
