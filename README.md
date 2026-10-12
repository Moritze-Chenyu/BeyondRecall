<div align="center">
  <img src="assets/readme/beyond_recall_banner.png" alt="Beyond Recall" width="960" />
  <h2>A Fine-Grained Multimodal Benchmark for Long-Term Agent Memory</h2>
  <p>
    <a href="START_HERE.md">Quick Start</a> &middot;
    <a href="data/BeyondRecall_text/README.md">Text Dataset</a> &middot;
    <a href="METHODS.md">Methods</a> &middot;
    <a href="CONFIGURATION.md">Configuration</a> &middot;
    <a href="CITATION.md">Citation</a>
  </p>
  <p><strong>11 Tasks</strong> &nbsp; | &nbsp; <strong>3 Memory Dimensions</strong> &nbsp; | &nbsp; <strong>1,269 QA Pairs</strong> &nbsp; | &nbsp; <strong>23,775 Sessions</strong></p>
</div>

Beyond Recall evaluates how agent memory systems retrieve, reason about and manage
information across long conversations with images, videos and documents.

## Highlights

- **Fine-grained memory abilities:** 11 tasks across retrieval, reasoning and management.
- **Multimodal evidence across sessions:** questions draw on images, videos and documents encountered at different times.
- **Long-term interactions:** 1,163 conversation trajectories, 1,269 QA pairs and 23,775 timestamped sessions.
- **A shared evaluation interface:** common input preparation and question-level state isolation across supported methods.
- **Time-aware evaluation:** question timestamps filter available history before memory ingestion.

## Task Examples

<p align="center">
  <a href="assets/readme/task_examples.png">
    <img src="assets/readme/task_examples.png" alt="Examples of all 11 Beyond Recall tasks, grouped into memory retrieval, reasoning and management" width="1000" />
  </a>
</p>

<p align="center"><em>Examples from the paper: task-relevant multimodal evidence and questions across the three memory dimensions. Click the figure to view its original resolution.</em></p>

This repository contains English benchmark text, questions, reference answers,
timestamps, media references, evaluation and framework source code, environment
specifications, launch instructions, and the documentation graphics above.
Benchmark images, videos, PDFs, model weights, credentials, databases and runtime
logs are distributed separately or generated locally. See [START_HERE.md](START_HERE.md)
to run an evaluation.

## Repository layout

| Directory | Contents |
|---|---|
| `data/BeyondRecall_text/` | 11 task files: 1,163 trajectories, 1,269 QA pairs, 23,775 sessions and 345,738 turns |
| `src/`, `adapters/` | Unified input, temporal filtering, retrieval, answering and judging |
| `third_party/` | Tested Mem0 OSS, A-Mem and MIRIX runtime source, licenses and compatibility patches |
| `config/`, `examples/` | API/local examples and an environment-based API configuration template |
| `requirements/`, `environments/` | Dependency versions for separate framework environments |
| `scripts/` | Dataset assembly, preprocessing, framework inference and service startup |
| `assets/readme/` | Title artwork and the paper task-example figure for this README |

## Methods and inputs

| Profile | Implementation | Representation |
|---|---|---|
| `mem0_agentic` | Legacy non-graph Mem0-A: extraction, action-based consolidation and vector retrieval | Text and frozen captions |
| `mem0_oss` | Newer Mem0-O: additive extraction and entity/lexical/vector retrieval | Text and frozen captions |
| `amem` | Actual A-Mem SDK notes, evolution and agentic search | Text and frozen captions |
| `rag_dense`, `rag_hybrid` | FastEmbed retrieval; optional BM25/RRF fusion | Text and frozen captions |
| `mirix` | Actual MIRIX service with synchronous, isolated writes | Native media fragments and aligned audio transcripts |
| `full_context_multimodal`, `rag_bm25_native` | Full visible context or BM25-selected complete sessions | Native media |

`mem0_agentic` and `mem0_oss` are distinct built-in methods using separate verified
SDK snapshots. Run them in separate processes; see [MEM0_VARIANTS.md](MEM0_VARIANTS.md). `multimodal_rag`, `oracle_caption`, `oracle_multimodal`, `text_only_shortcut` and
`question_only` require external adapters and cannot run with `adapters/builtin.py`.
See [METHODS.md](METHODS.md) for every profile's implementation status and prerequisites. Third-party snapshots and their changes are described in
[THIRD_PARTY.md](THIRD_PARTY.md). API and local answer backends share the interface.

Question timestamps filter history **before ingestion**. Native videos retain all
frames and their original audio tracks in stream-copy clips, paired with aligned
ASR text. At a text/image API boundary, every decoded video frame is delivered as
lossless tiles, together with the transcript. Current release PDFs retain all
annotated evidence pages and boundary pages; every retained page is rendered at
300 DPI when required by a vision API. The evaluator does not silently remove
pages, sample frames, or truncate input to resolve capacity errors. Provider-side
processing remains provider-dependent. Read [INPUT_PROTOCOL.md](INPUT_PROTOCOL.md)
for the full contract.

## Data

The [text release](data/BeyondRecall_text/README.md) preserves the benchmark JSON
records and original media paths. It is not a complete multimodal dataset on its
own. Download the corresponding media release from Hugging Face and assemble the
verified dataset before evaluating; the launch guide explains this process.
A non-Latin character inside a vehicle identifier is Unicode-escaped in the public
JSON, preserving its exact decoded value.

Experimental settings are described in the paper. See [CONFIGURATION.md](CONFIGURATION.md)
for configuration fields and supported input modes. Verify the repository file
inventory with `sha256sum -c SHA256SUMS`.

## Citation

Please cite the accompanying paper when using this benchmark or evaluation code.
See [CITATION.md](CITATION.md), [CITATION.cff](CITATION.cff) and
[CITATION.bib](CITATION.bib). The current anonymous manuscript citation is
provisional; actual authors and arXiv metadata must be filled in after release.

## Licensing

The license decision for original code and benchmark contributions is pending.
[LICENSE](LICENSE) currently records that status; [LICENSING.md](LICENSING.md)
defines the proposed code/data scope. Third-party SDKs retain their own licenses;
upstream conversation text and separately distributed media are not automatically
relicensed.
