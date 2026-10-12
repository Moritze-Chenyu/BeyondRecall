#!/usr/bin/env python3
"""Keep evidence + first/last pages; optionally allow mandatory pages above 10."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from beyond_recall_eval.pdf_evidence import select_evidence_pages
from beyond_recall_eval.pdf_hybrid import export_core_pdf, extract_pages
from beyond_recall_eval.media import CapacityError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--page-limit', type=int, default=10)
    parser.add_argument('--allow-evidence-overflow', action='store_true')
    parser.add_argument('--verify-rendered-pages', action='store_true',
                        help='Compare every retained page pixel-for-pixel and check for hidden page objects.')
    args = parser.parse_args()
    if args.dataset.resolve() == args.output_dir.resolve() or args.dataset.resolve() in args.output_dir.resolve().parents:
        raise ValueError('Output must be a separate derived directory outside the source dataset.')
    provenance = json.loads(args.provenance.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared, conflicts, errors, plans, cached = [], [], [], {}, {}
    for item in provenance['matches']:
        source = args.dataset / item['release_pdf']
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Source PDF no longer matches the audited source dataset.')
        texts, _ = extract_pages(data)
        count = len(texts)
        evidence = item['all_source_evidence_pages']
        mandatory = sorted(set(evidence) | {1, count})
        row = {'release_pdf': item['release_pdf'], 'source_sha256': digest,
               'source_doc_ids': item['source_doc_ids'], 'original_pages': count,
               'evidence_pages': evidence, 'mandatory_pages': mandatory,
               'evidence_scope': 'union of all source QA annotations for this document; conservative fixed document view'}
        try:
            pages, reasons = select_evidence_pages(count, evidence, args.page_limit, args.allow_evidence_overflow)
        except CapacityError as error:
            conflicts.append({**row, 'minimum_required_pages': len(mandatory), 'reason': str(error)})
            print('CONFLICT', item['release_pdf'], len(mandatory), flush=True)
            continue
        except ValueError as error:
            errors.append({**row, 'error_type': 'InvalidSourcePageAnnotation',
                           'invalid_page_numbers': [p for p in evidence if not 1 <= p <= count],
                           'reason': str(error)})
            print('ANNOTATION_ERROR', item['release_pdf'], count, errors[-1]['invalid_page_numbers'], flush=True)
            continue
        destination = args.output_dir / item['release_pdf']
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if digest in cached:
                destination.write_bytes(cached[digest])
            else:
                export_core_pdf(data, pages, destination)
                cached[digest] = destination.read_bytes()
            # Independently read the generated PDF, not only the manifest.
            actual, _ = extract_pages(destination.read_bytes())
            assert len(actual) == len(pages)
            assert len(pages) <= args.page_limit or (args.allow_evidence_overflow and pages == mandatory)
            for new, original in zip(actual, pages):
                assert ''.join(new.split()) == ''.join(texts[original - 1].split())
            assert set(evidence).issubset(pages) and 1 in pages and count in pages
            if args.verify_rendered_pages:
                import fitz
                with fitz.open(stream=data, filetype='pdf') as original, fitz.open(destination) as cropped:
                    for index, number in enumerate(pages):
                        before = original[number - 1].get_pixmap()
                        after = cropped[index].get_pixmap()
                        assert (before.width, before.height, before.samples) == (after.width, after.height, after.samples)
                    assert sum(cropped.xref_get_key(x, 'Type') == ('name', '/Page')
                               for x in range(1, cropped.xref_length())) == len(pages)
            row.update(selected_original_pages=pages, output_pages=len(actual), selection_reasons=reasons,
                       mandatory_evidence_overflow=len(pages) > args.page_limit,
                       rendered_pages_verified=args.verify_rendered_pages,
                       output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            prepared.append(row)
            plans[digest] = pages
            destination.with_suffix('.audit.json').write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
        except Exception as error:
            destination.unlink(missing_ok=True)
            errors.append({**row, 'error_type': type(error).__name__, 'reason': str(error),
                           'tool_stderr': getattr(error, 'stderr', b'').decode(errors='replace') if isinstance(getattr(error, 'stderr', b''), bytes) else str(getattr(error, 'stderr', ''))})
    report = {'mode': 'source_evidence_opening_closing_other_pages', 'page_limit': args.page_limit,
              'allow_evidence_overflow': args.allow_evidence_overflow,
              'source_pdfs_modified': False, 'text_memory_policy': 'not decided by this crop-only export',
              'source_files': len(provenance['matches']), 'prepared_files': len(prepared),
              'conflict_files': len(conflicts), 'error_files': len(errors),
              'prepared': prepared, 'conflicts': conflicts, 'errors': errors}
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'fixed_pdf_pages.json').write_text(json.dumps(plans, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'prepared', 'conflicts', 'errors'}}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
