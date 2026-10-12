# Launch guide

See [METHODS.md](METHODS.md) before selecting a profile: some names are
protocol-only and require external adapters. Mem0 Agentic and OSS both have
built-in adapters using separate SDKs; see [MEM0_VARIANTS.md](MEM0_VARIANTS.md).

Use Python 3.11 and separate environments for A-MEM and Mem0/MIRIX/RAG.
Commands below run from the repository root. Experimental settings are described
in the paper; configuration fields are documented in `CONFIGURATION.md`.

## 1. Install an environment

```bash
# Mem0 OSS, MIRIX and dense/hybrid RAG:
python -m pip install -r requirements/mem0_mirix.txt
python -m pip install --no-deps -e frameworks/Mem0
python -m pip install --no-deps -e frameworks/MIRIX
python -m pip install --no-deps -e .

# Mem0 Agentic, in a separate environment:
python -m pip install -r requirements/mem0_mirix.txt
python -m pip install --no-deps -e frameworks/Mem0-Agentic
python -m pip install --no-deps -e .

# A-MEM, in a separate environment:
python -m pip install -r requirements/amem.txt
python -m pip install --no-deps -e frameworks/A-mem
python -m pip install --no-deps -e .
```

`--no-deps` preserves the recorded dependency versions. Install MIRIX in editable
mode because its upstream sdist excludes server modules. The framework launcher
selects bundled source and verifies its SHA256 snapshot without requiring Git history.

For the local answer backend, use a separate environment with
`python -m pip install -e '.[local,media]'`, configure
`config/local_pdf_cropped_10.json`, and use the general CLI.
Embedding models are downloaded on first use unless already cached.

## 2. Assemble text and media

Obtain the matching media release. Its root must contain the `images/` and
`videos/` paths referenced by the task JSON. Use the source release described by
`data/benchmark/SOURCE_SHA256SUMS`.

```bash
python scripts/assemble_dataset.py \
  --media-root /path/to/downloaded/media-release \
  --output results/dataset
```

Assembly verifies source checksums, restores source task-file bytes and preserves
the release fingerprint. It refuses to overwrite an existing output. On the same
filesystem, `--mode hardlink` avoids duplicating media bytes; do not edit linked
inputs. Native evaluation requires the assembled dataset and matching media.

## 3. Configure APIs and prepare inputs

Copy `config/.env.example` to `.env`, fill in the provider settings, and export them:

```bash
test -f .env || cp config/.env.example .env
# Fill in provider settings in .env before exporting them.
set -a
source .env
set +a
```

The framework launcher reads `examples/api_config.py` by default.
`--previous-config /path/to/your_config.py` selects an explicit compatible
configuration. Optional empty settings fall back to the primary API settings.
Provider model names, thinking controls and context limits must match the deployment.
`.env`, `results/`, `media/`, `runtime_assets/` and model caches are ignored by Git.

For caption methods, supply a frozen caption cache at
`runtime_assets/captions.json`, or pass `--caption-store`. Its hashes must match
current media and its coverage must include all selected inputs.
`scripts/import_legacy_captions.py` imports existing captions from explicitly
supplied historical sources; `scripts/complete_legacy_captions.py` supplements
missing image and PDF captions. Supplementation does not generate video captions
from scratch; retain the producer and prompt provenance for any regenerated cache.

For MIRIX, prepare original-quality video clips and aligned ASR transcripts:

```bash
python scripts/prepare_video_segments.py \
  --dataset results/dataset \
  --output-dir runtime_assets/video_segments --segment-seconds 10
```

This calls the configured ASR API and preserves all source frames and audio packets.
To prepare only a subset, pass an explicit `--sample-manifest`.

## 4. Run framework inference

```bash
python scripts/run_framework.py \
  --dataset results/dataset --profile mem0_oss \
  --caption-store runtime_assets/captions.json \
  --output-dir results/mem0 --workers 2
```

Change `--profile` to `mem0_agentic`, `amem`, `rag_dense` or `rag_hybrid` in the appropriate
environment. The launcher uses all questions by default. For a subset, create a
selection with `scripts/select_sample.py` and pass `--sample-manifest` explicitly.
Each question receives a fresh namespace and timestamp-filtered history.
The output includes predictions, the resolved configuration, a dependency lock
and an inference manifest. Inference completion counts are not accuracy scores.

## 5. Run MIRIX

Start a dedicated MIRIX service:

```bash
PYTHONPATH=frameworks/MIRIX python scripts/serve_mirix.py \
  --port 8540 --state-dir results/mirix_service_state \
  --llm-extra-body '{"enable_thinking":false}' \
  --empty-tool-confirmation --llm-request-timeout 300
```

In a second terminal in the same environment:

```bash
MIRIX_BASE_URL=http://127.0.0.1:8540 \
BR_MIRIX_MEDIA_REGISTRATION_PROTOCOL=async_inline_bytes_v1 \
BR_MIRIX_EMPTY_TOOL_CONFIRMATION=1 \
python scripts/run_framework.py \
  --dataset results/dataset --profile mirix \
  --mirix-memory-model "$BR_MEMORY_MODEL" \
  --video-segment-manifest runtime_assets/video_segments/manifest.json \
  --output-dir results/mirix --workers 2
```

`BR_MULTIMODAL_MODEL` selects the answer model and `--mirix-memory-model` selects
the memory model. Use a separate state directory and port for independent services.
Capacity and provider errors remain recorded without reducing source inputs.

## 6. Judge answers and calculate metrics

The general CLI supports inference, separate judging and metric aggregation:

```bash
python -m beyond_recall_eval.cli infer --help
python -m beyond_recall_eval.cli judge --help
python -m beyond_recall_eval.cli summarize --help
```

For example, judge and summarize the Mem0 predictions:

```bash
python -m beyond_recall_eval.cli judge \
  --config results/mem0/configuration.json \
  --predictions results/mem0/predictions.jsonl \
  --inference-manifest results/mem0/predictions.run.json \
  --output results/mem0/judgments.jsonl

python -m beyond_recall_eval.cli summarize \
  --dataset results/dataset --config results/mem0/configuration.json \
  --profile mem0_oss --predictions results/mem0/predictions.jsonl \
  --judgments results/mem0/judgments.jsonl \
  --inference-manifest results/mem0/predictions.run.json \
  --judge-manifest results/mem0/judgments.run.json \
  --output results/mem0/metrics.json
```

Configure the judge model and its environment variables before judging.
For subset inference, pass the identical `--sample-manifest` to `summarize`.
For general CLI inference, supply an adapter, framework version and dependency
lock as described by `infer --help`.
