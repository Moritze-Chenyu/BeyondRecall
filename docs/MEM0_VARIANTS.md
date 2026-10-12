# Mem0 Agentic and OSS

In this benchmark, `mem0_agentic` identifies Mem0-A, the legacy **non-graph**
framework. It does not identify a hosted Mem0 service or A-MEM. `mem0_oss`
identifies Mem0-O, the newer mainline framework. This terminology follows the
paper and the historical ordinary versus `--oss-mainline` experiment paths.

| Property | Mem0 Agentic / Mem0-A | Mem0 OSS / Mem0-O |
|---|---|---|
| Profile | `mem0_agentic` | `mem0_oss` |
| Implementation | `Mem0AgenticSystem` | `Mem0OSSSystem` |
| Bundled source | `frameworks/Mem0-Agentic/` | `frameworks/Mem0/` |
| SDK | Legacy non-graph 1.0.6 | Newer mainline 2.0.2 with recorded local changes |
| Extraction | Fact-list extraction | Additive memory-object extraction |
| Consolidation | ADD / UPDATE / DELETE / NONE decisions | Additive writes with hash deduplication |
| Retrieval call | `search(query, user_id=..., limit=...)` | `search(query, filters={user_id: ...}, top_k=...)` |
| Retrieval algorithm | Legacy vector retrieval | Newer entity, lexical and vector scoring |

Both adapters use complete frozen captions, the same timestamp-visible prepared
history, private question namespaces, provenance metadata and the configured
answer backend. Graph memory is disabled for these profiles. They share adapter
mechanics while preserving the algorithms implemented by each SDK. JSON contracts
are validated according to the selected algorithm before the SDK parses them.

## Run Mem0 Agentic

Use its own Python 3.11 environment and run from the repository root:

```bash
python -m pip install -r requirements/mem0_mirix.txt
python -m pip install --no-deps -e frameworks/Mem0-Agentic
python -m pip install --no-deps -e .
test -f .env || cp config/.env.example .env
# Fill in provider settings in .env before exporting them.
set -a
source .env
set +a
python scripts/run_framework.py \
  --dataset /path/to/assembled/dataset --profile mem0_agentic \
  --caption-store /path/to/frozen/captions.json \
  --output-dir results/mem0_agentic --workers 2
```

For OSS, install `frameworks/Mem0` in a separate environment and use
`--profile mem0_oss` with a different output directory. The launcher chooses the
right bundled source automatically, verifies its hashes, and records the source
version and commit. A different SDK already imported into a process causes a
clear error; modules are never swapped while live memory objects exist.
`--framework-source` can select an explicit snapshot containing `SDK_SOURCE.json`;
its version must match the profile. The general CLI also supports both variants.
For its SDK configuration, export `BR_FRAME_MEMORY_SDK_URL` as a base URL ending
in `/v1` or the provider's equivalent base path, without `/chat/completions`.

The newer SDK optionally uses spaCy for entity extraction and lemmatization.
Install the appropriate spaCy English model when reproducing that configuration;
without it, the upstream SDK uses its documented fallback. Record that choice
alongside the source version. Do not equate different snapshots or NLP settings
with exact reproduction of the paper experiments.

## Validation status

The earlier 20-question run labeled OSS used the installed legacy 1.0.6 SDK.
It therefore cannot validate the newly integrated mainline 2.0.2 implementation.
Current integration checks use separate processes, real SDK/Qdrant state and
scripted LLM/embedding fixtures. These checks establish code behavior and source
separation; they are not real-provider inference or model accuracy measurements.

Both variant integrations passed the same frozen 20-question selection across
all 11 tasks, including four timestamped questions. Each run ingested 297 visible
sessions and excluded 29 future sessions, checked source lineage and cleanup,
and rejected cross-loading of the other SDK. Tests used deterministic LLM and
embedding fixtures with the actual SDK and local Qdrant, not real provider calls.
Fourteen framework regression checks also passed against the public source.

Run the portable fixture checks without API credentials:

```bash
python scripts/check_mem0_variants.py --output-dir results/mem0_sdk_checks
```

To additionally check a frozen benchmark selection with real prepared inputs:

```bash
python scripts/check_mem0_variants.py \
  --dataset /path/to/assembled/dataset \
  --sample-manifest /path/to/sample.json \
  --caption-store /path/to/frozen/captions.json \
  --output-dir results/mem0_sdk_checks
```

These commands substitute deterministic testing transport and embeddings. They
never contact an LLM service, and their saved outputs are engineering reports.
They are separate from `run_framework.py`, which runs the configured real models.
