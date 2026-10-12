#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from beyond_recall_eval.dataset import BeyondRecallDataset
from beyond_recall_eval.sampling import build_sample

parser = argparse.ArgumentParser(description='Freeze a task- and modality-stratified evaluation subset.')
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--count', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
sample = build_sample(list(BeyondRecallDataset(args.dataset).trajectories()), args.dataset, args.count, args.seed)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: value for key, value in sample.items() if key not in {'items', 'question_keys'}}, indent=2))
