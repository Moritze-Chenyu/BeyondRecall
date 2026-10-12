# Benchmark Dataset

This directory contains the complete benchmark conversation text and QA pairs:
1,163 trajectories, 1,269 questions, 23,775 sessions and 345,738 turns across
11 tasks in retrieval, reasoning and management. Task JSON preserves IDs,
participants, session/turn timestamps, question cutoffs, reference answers and
relative media paths. Decoded records are identical to the current source release.

No image, PDF, spreadsheet, video or audio file is included here. Native evaluation
requires the separately released media. Use `scripts/assemble_dataset.py` from the
repository root to combine text with the matching media and verify all checksums.
The source checksum list is named `SOURCE_SHA256SUMS` to distinguish it from the
text-only file inventory; many of its files intentionally live in the media release.

All prose is English. The original non-Latin character inside a vehicle identifier
is JSON Unicode-escaped to preserve the benchmark value. `TEXT_RELEASE.json`
records the exact byte restoration needed for frozen dataset fingerprints.
`SOURCE_README.md`, `RELEASE_STATS.json` and `PDF_PREPROCESSING.json` retain the
source release metadata and PDF page policy. Annotated evidence pages are used
only for fixed dataset preprocessing, not query-specific memory filtering.
