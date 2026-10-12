# Fixed evidence-page PDF policy

The current media release retains source-annotated evidence pages and the first
and last page of each PDF. When these mandatory pages fit within ten pages,
additional source pages fill the remaining budget up to ten. If mandatory pages
exceed ten, every mandatory page remains and no filler pages are added. Thus ten
is a target, not an absolute page limit.

The published media contains 93 PDFs: 92 were replaced by verified crops and one
remains original because source evidence annotations are invalid. Forty cropped
PDFs exceed the ten-page target. Discarded pages are not passed to memory as text
in this release. Per-file original page mappings and status are recorded in
`data/benchmark/PDF_PREPROCESSING.json`.

Selection was fixed during dataset preparation, conservatively using the union
of source-dataset annotations for each document. The evaluator does not select
pages dynamically from a test question or its answer. Every retained page is
provided when a native framework consumes the corresponding document. Vision-only
model boundaries render all retained pages at 300 DPI, without a further page cap.
Earlier results using full source PDFs are not directly comparable to this
revised release; run manifests must identify the exact input version.
