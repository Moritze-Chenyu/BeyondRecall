# Beyond Recall Dataset

This directory contains the anonymized release of Beyond Recall.

## Structure

- `retrieval/`, `reasoning/`, and `management/` contain the 11 task files.
- `images/<task>/` contains locally packaged images and PDF documents.
- `videos/<task>/` contains locally packaged video clips.
- Every path stored in a task file is relative to this directory.

## Record Schema

Each task file is a JSON array. Record IDs start from `0` independently in
each task file.

```json
{
  "id": 0,
  "persons": ["Person A", "Person B"],
  "conversation": [
    {
      "date": "timestamp",
      "turns": [
        {
          "person": "Person A",
          "content": "dialogue text",
          "media": ["images/task/0.jpg"]
        }
      ]
    }
  ],
  "questions": [
    {
      "question": "question text",
      "answer": "reference answer",
      "media": ["images/task/1.pdf"]
    }
  ]
}
```

The optional `media` field is present only when a turn or question has a
packaged image, PDF document, or video. `persons` lists the two primary human
participants; it is not an exhaustive role vocabulary. Context adapted from
MMDU may include `assistant` turns, which are preserved as part of the
chronological interaction history.

## Statistics

- Conversation trajectories: 1163
- Question-answer pairs: 1269
- Timestamped sessions: 23775
- Dialogue turns: 345738
- Task categories: 11
- Unique source images: 1865
- Unique source PDF documents: 93
- Unique source videos: 314

Carrier-conversation image URLs and construction metadata are not part of
this release. They are unrelated to the annotated answer evidence.

## PDF preprocessing

92 of the 93 packaged PDF files were replaced with verified crops.
Each crop retains all source-annotated evidence pages and first/last pages.
When those mandatory pages fit within 10 pages, other pages are added up to 10.
When mandatory pages exceed 10, all mandatory pages are retained, with no extras.
Discarded pages are not provided as text. Evidence selection conservatively
uses the union of source-dataset QA annotations for each document.

40 cropped PDFs exceed the 10-page target to preserve mandatory evidence.
1 file remains original because source evidence annotations are invalid.
The release therefore does not impose a universal 10-page PDF limit.
See `PDF_PREPROCESSING.json` for per-file status and original-page mappings.
Task JSON, question/answer text, timestamps, media paths and file counts
are unchanged. This is a revised dataset input; earlier full-PDF results
are not directly comparable.
