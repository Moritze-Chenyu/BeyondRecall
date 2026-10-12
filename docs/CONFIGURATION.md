# Configuration and fidelity

See [METHODS.md](METHODS.md) before selecting a profile: some names are
protocol-only and require external adapters. Mem0 Agentic and OSS both have
built-in adapters using separate SDKs; see [MEM0_VARIANTS.md](MEM0_VARIANTS.md).

Experimental settings are described in the paper. The JSON files in `config/`
provide API/local configuration templates; set model IDs, revisions and provider
parameters for your deployment. `evaluation_protocol.json` retains existing
profile names for adapter compatibility. The run manifest hashes code,
configuration, selected questions, adapter, environment lock and dataset release
checksum list. Credentials are resolved from environment variables; provider URLs
are hashed in the manifest.

## Common fields

- `models.answer.backend`: `api` or `transformers`; judge uses the same backend
  choices independently. Other model roles are used only by relevant adapters.
- `models.<role>.model`, `revision`, `revision_env`, `chat_template_revision`:
  actual model ID, snapshot and template. Local loading honors `revision_env`.
- `method_profiles.<name>.implementation`: `full_context`, `bm25`, `mem0_caption`,
  `amem_caption`, `dense_rag`, `mirix_native` for builtin adapters. Framework SDKs
  and model/service configuration remain explicit dependencies.
- `media_processing.quality.pdf_mode`: `core_pdf_remaining_text` in the updated API/local
  examples. Before memory ingestion it creates a core PDF file and texts for
  only the other pages. `core_visual_full_text` and `all_pages` remain legacy
  alternatives; they are not the current memory-input format. It is an explicitly
  information-reduced visual protocol and must be fixed across compared methods.
- `pdf_visual_pages`: default 10 per document, fewer for short PDFs. Selection is
  deterministic and document-only; review it before published experiments.
- `pdf_page_overrides`: optional mapping from opaque asset ID to original page
  numbers, e.g. `{ "media_example": [1, 2, 5, 9] }`. No duplicates; at most
  the visual budget. Do not select pages based on reference answers.
- `pdf_ocr_missing_text`: default true. Tesseract processes textless pages at
  300+ DPI with language `pdf_ocr_language` (default `eng`). Configure appropriate
  language data for the dataset. Low-confidence text is preserved and flagged
  for review. Without OCR, all textless pages must fit in the visual budget,
  otherwise the PDF fails rather than producing an empty-text memory.
- `media_processing.quality.pdf_render_dpi`: at least 300, all selected visual pages. Higher DPI
  costs more tokens. Set higher DPI if fine visual detail requires it; rendering
  at 300 DPI is not mathematically lossless for vector graphics.
- `native_file_types`: only formats proven supported by the actual endpoint.
  A chat-completions API accepting text/images does not imply it accepts files.
- `video_transport`: `unsupported`, `native_file`, `video_url`, `all_frames`.
  Select a request format supported by the endpoint. Provider-internal processing
  is outside the public evaluator's acceptance requirements.
- `spreadsheet_mode`: `native_required` or `complete_xml_and_render`. Expanded
  mode requires XLSX-compatible content and working LibreOffice. XLS/ODS and
  interactive/macro content require native support or a dedicated adapter.
- `maximum_pdf_pages`: optional all-pages-mode expansion budget; over-budget documents fail
  as a whole, with no partial-page result. It is not applied to hybrid core-page mode, which represents core pages as PDF and only the other pages as text.
  Limiting image count does not guarantee fitting the model context.
- `maximum_source_bytes` and `models.answer.maximum_request_bytes`: optional
  capacity guards. Oversized inputs fail with no shrinking fallback.
- Local `tile_size`: multiple of model patch/merge factor (1008 for the example).
  Padding is white; original pixels are not interpolated. `context_window` must
  match the deployed model. It is not permission to enlarge its true capacity.

## Optional hosted processor audit

Public API configurations default to `require_verified_processing: false`.
They require supported requests, complete application-side inputs, correct
timestamp filtering, and explicit capacity/error handling. No provider certificate
or successful marker-recognition probe is required to run evaluation. Record the
actual provider/model settings; provider-specific preprocessing may differ.

For a separate deployment audit, optionally set `require_verified_processing: true`
and provide `provider_processing_report` as a JSON object:

```json
{
  "model_revision": "actual immutable deployment revision",
  "processor_revision": "actual server processor commit",
  "audit_artifact_sha256": "64 lowercase hexadecimal characters of the real audit artifact",
  "no_input_truncation": true,
  "pixel_preserving_images": true,
  "complete_file_processing": true
}
```

These are attestations backed by an actual server audit, not switches that make
a hosted processor lossless. Inspect the server preprocessing code and test the
full content: original image pixel coverage, every document page and workbook
component, video frames and audio, and tokenizer/context limits. Keep the artifact
alongside experiment records. This client validates the report's structure and
records it; it cannot independently authenticate the server implementation.
Pure text calls do not require this media report. Provider-specific thinking
controls belong in `models.answer.extra_body`; a configuration label alone does
not prove that a service disabled its internal reasoning. This optional audit
does not change the public benchmark's default acceptance requirements.

## Captions and framework state

Caption profiles require `--caption-store` containing a frozen, complete cache
with the expected producer/prompt fingerprint. Native media evaluations must not
silently substitute those captions. Mem0 requires explicit `framework_config`;
its memory-extraction and embedding model configuration must also be frozen.
The answer backend can be local even when the memory framework uses an API.
Different framework implementations may transform/retrieve memory differently;
all receive exactly the same timestamp-visible source history.

### Real framework settings

Install the appropriate bundled Mem0 SDK from `START_HERE.md`; `.[mem0]` supplies
common dependencies without choosing a variant. Install `.[rag]` for dense/hybrid RAG. A-Mem also needs
`.[amem]` and its external SDK checkout on `PYTHONPATH`; MIRIX needs its SDK and
a running MIRIX service. Do not install incompatible SDK dependency sets into
one environment. The dependency files record versions for separate framework environments.

`framework_config` resolves keys ending in `_env` only at runtime. For example,
`api_key_env: "MEMORY_API_KEY"` becomes an SDK `api_key`; configuration files
contain environment variable names, never credentials.

- Mem0: provide SDK `llm`, `embedder` and Qdrant `vector_store` configuration.
  The LLM model must equal `models.memory_extraction_and_evolution.model`.
  The adapter creates its own private Qdrant directory, collection and history
  database. JSON contract failures receive bounded repair retries; SDK errors
  that upstream catches internally become explicit evaluation failures.
- A-Mem: `framework_config.embedding_model`, `metadata_mode` (`simple` or `llm`)
  and `evolution_mode: "llm"`. Simple note metadata can be combined with LLM
  evolution and SDK agentic retrieval. Full session text
  and timestamps remain in each note. The SDK's default constructor storage
  is replaced with question-scoped storage; note/evolution/search logic is used
  directly.
  The adapter resolves displayed neighbor ranks through actual Chroma result IDs,
  so graph links and neighbor updates address the intended notes. Consolidation
  uses upsert for existing IDs, and evolved metadata accompanies retrieved text.
- Dense RAG: `embedding_role: "text_embedding"`; set its real encoder model.
  Hybrid RAG uses both `dense_candidates` and `lexical_candidates` before
  final `retrieval_top_k`, with reciprocal-rank fusion using `rrf_k`. The
  BGE-small encoder has its own 512-token embedding limit; full selected session
  text is stored and delivered to the answer model. This is a retrieval encoder
  limit, not a truncation of model-visible source material.
- MIRIX: provide `base_url_env`, `timeout_seconds`, `org_id` and `agent_config`.
  Agent names must be the actual SDK names, e.g. `core_memory_agent`. Seed core
  blocks through the agent list entry
  `{"core_memory_agent":{"blocks":[{"label":"human","value":""},
  {"label":"persona","value":"I am a helpful assistant."}]}}`.
  A nested `meta_agent_config.memory` field is ignored by the bundled service.
  All retained PDF pages and all video frames are expanded at the service model
  boundary; native handles remain in the prepared ingestion interface. Calls
  await synchronous processing; no caption substitution is performed. Cleanup
  deletes both user memories and that question's client-owned templates.
  `framework_config.retrieval_limit_per_type` limits each memory group separately
  (10 in the launcher). The global `retrieval_top_k` budget is 120; all returned
  recent/relevant memories and core blocks are delivered, with duplicates removed.
  Exceeding that budget fails explicitly instead of dropping later memory groups.

The bundled MIRIX checkout has a synchronous image-registration helper that can
block an async SQLite request. `scripts/serve_mirix.py` runs the actual
SDK app on an isolated port and SQLite directory, with asynchronous registration
of the exact inline image bytes. It also retains query images during topic
extraction and supports explicit provider thinking controls through the OpenAI
SDK's `extra_body`. No note, memory evolution or retrieval algorithm is replaced.
Use a dedicated environment with the real MIRIX checkout on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/MIRIX python scripts/serve_mirix.py \
  --port 8533 --state-dir results/mirix_service/state \
  --llm-extra-body '{"enable_thinking":false}'

MIRIX_BASE_URL=http://127.0.0.1:8533 \
BR_MIRIX_MEDIA_REGISTRATION_PROTOCOL=async_inline_bytes_v1 \
python scripts/run_framework.py --dataset results/dataset \
  --profile mirix --framework-source /path/to/MIRIX \
  --output-dir results/mirix --workers 4
```

The adapter verifies the service protocol and thinking controls before ingestion.
The service limits concurrent LLM requests with `--llm-concurrency`
(default 2). Rate-limit retries resend the same complete request, with a default
65-second delay and at most six attempts; media and text are never reduced to fit quotas.
Each provider call has a configurable `--llm-request-timeout` (default 300 seconds).
If a memory-model call fails, the service cancels the request's remaining SDK
calls and returns an explicit error, even if the upstream tool handler would
otherwise catch it and report partial processing as successful.
The adapter disables SDK HTTP retries: memory writes are not idempotent, and
retrying a partially completed write can duplicate memories. Rate-limit retries
are confined to individual model calls before the SDK receives a completion.
Question-owned media metadata and temporary registered files are cleaned after
memory deletion. Do not point this service at another experiment's
database. The source checkout, service-wrapper SHA and actual dependency versions
are frozen in its lock file.

Some OpenAI-compatible providers stall when a required tool has an empty object
schema. The service offers explicit `--empty-tool-confirmation`
compatibility for `finish_memory_update`; declare it to the launcher with
`BR_MIRIX_EMPTY_TOOL_CONFIRMATION=1`. The provider receives a required Boolean
acknowledgement. Only a real `{"acknowledged":true}` tool response is converted
back to the SDK's original `{}` arguments; invalid confirmations fail. The model
still selects and executes the SDK finish tool. This is a recorded transport
adaptation, not a fabricated completion or a change to the memory algorithm.

`scripts/run_framework.py` runs framework inference using exported API settings
or an explicitly supplied `--previous-config`. `--sample-manifest` selects a
frozen subset; omit it to use the full dataset. The launcher saves its resolved
configuration and dependency lock alongside predictions. Use those same files
for separate judging and metric aggregation through the general CLI.

Each question ingests fresh visible history, so earlier queries cannot leak later
state. A timestamp date without an explicit zone is interpreted in the dataset's
common timeline (normalized as UTC); do not combine datasets with inconsistent
implicit zones. Timestamp comparisons are inclusive. Timing is imposed by the
runner, before staging/caption lookup/framework construction.

Capacity failures, unsupported files and API failures are retained as errors;
strict metrics count them as incorrect. A subset run must use the same sample
manifest for inference and summary. The output identifies representative subsets
and does not mark them as formal full-dataset results.

Tesseract setup and output formats follow its [official installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html) and [command-line documentation](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html). OCR is not a proof of lossless information conversion.
