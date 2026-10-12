# Input and temporal protocol

All frameworks implement the same ingest/query/close contract. A question's
cutoff is applied before any preparation or memory write. Timestamped turns and
sessions are processed chronologically, with the cutoff inclusive. Each question
has an isolated namespace; cleanup must remove its state.

## Native multimodal methods

Text is not truncated. Original images and file handles enter memory preparation
with source IDs. The current published PDFs were preprocessed using annotated
evidence pages, first/last pages and optional filler pages up to ten. If mandatory
pages exceed ten, all mandatory pages remain. Every retained page enters the
vision boundary at 300 DPI. This revised input differs from older full-PDF runs;
see `data/benchmark/PDF_PREPROCESSING.json` and `PDF_EVIDENCE_POLICY.md`.

Video is split at existing keyframes with stream copy. Every original decoded
frame and audio packet is preserved; clips can exceed the target ten seconds.
Each clip is paired with transcription of its own audio interval and enters memory
as one fragment. At an image-based model boundary, every frame becomes lossless
PNG tiles with position and sequence metadata. No frame sampling, client resizing
or caption substitution is used to avoid input limits. ASR does not preserve all
non-speech acoustic information, and automatic transcripts can contain errors.

Excel requires either complete native support or complete workbook expansion,
including OOXML, hidden content, embedded images and rendered sheets. Unsupported
components fail explicitly. API-internal resizing is outside this evaluator's guarantees.

## Text-only methods

These preserve the historical text-plus-frozen-caption baseline. Caption coverage
and current source hashes are checked. Captions are reduced representations;
execution success does not establish that every visual fact survived captioning
or memory evolution. They are not substituted into native multimodal methods.

## Failures and provenance

Provider errors, incomplete writes, missing assets and over-capacity inputs are
recorded rather than converted into successful empty or degraded answers. The MIRIX adapter does not guarantee
original source-unit IDs in retrieved SDK output; source-level retrieval metrics
require a trace that preserves those identifiers.
