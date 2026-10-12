#!/usr/bin/env python3
"""Replace verified successful PDF crops, with an external backup and rollback."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from beyond_recall_eval.pdf_hybrid import extract_pages


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_copy(source, destination):
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--crops', required=True, type=Path)
    parser.add_argument('--backup', required=True, type=Path)
    parser.add_argument('--extend-preprocessing', action='store_true',
                        help='Merge this replacement into existing preprocessing metadata.')
    args = parser.parse_args()
    root, crops, backup = args.dataset.resolve(), args.crops.resolve(), args.backup.resolve()
    if root == backup or root in backup.parents or backup.exists():
        raise ValueError('Backup must be a new directory outside the dataset.')
    report = json.loads((crops / 'report.json').read_text())
    if len(report['prepared']) != report['prepared_files']:
        raise ValueError('Crop report count mismatch.')
    metadata_names = ['README.md', 'RELEASE_STATS.json', 'SHA256SUMS']
    new_metadata = root / 'PDF_PREPROCESSING.json'
    prior_metadata = json.loads(new_metadata.read_text()) if new_metadata.exists() else None
    if new_metadata.exists() and not args.extend_preprocessing:
        raise ValueError('PDF preprocessing metadata already exists; refusing repeated replacement.')
    if prior_metadata:
        metadata_names.append('PDF_PREPROCESSING.json')
    expected_pdf_count = len(list(root.rglob('*.pdf')))
    checksums = dict((name, sha) for sha, name in
                     (line.split('  ', 1) for line in (root / 'SHA256SUMS').read_text().splitlines()))
    rows = report['prepared'] + report['conflicts'] + report['errors']
    if len(rows) != report['source_files'] or len({x['release_pdf'] for x in rows}) != len(rows):
        raise ValueError('Duplicate or missing source PDF paths.')
    before = {}
    for row in rows:
        relative = Path(row['release_pdf'])
        source = (root / relative).resolve()
        if relative.is_absolute() or root not in source.parents or source.suffix != '.pdf':
            raise ValueError('Invalid release PDF path.')
        sha = digest(source)
        if sha != row['source_sha256'] or checksums.get(row['release_pdf']) != sha:
            raise ValueError(f'Source/checksum changed: {relative}')
        before[row['release_pdf']] = sha
    for row in report['prepared']:
        crop = crops / row['release_pdf']
        if digest(crop) != row['output_sha256']:
            raise ValueError(f'Crop changed: {crop}')
        pages, _ = extract_pages(crop.read_bytes())
        originals, _ = extract_pages((root / row['release_pdf']).read_bytes())
        mapping = row['selected_original_pages']
        mandatory = set(row['evidence_pages']) | {1, len(originals)}
        if not len(pages) == len(mapping) == row['output_pages']:
            raise ValueError('Crop page count mismatch.')
        if len(pages) > report['page_limit'] and (not report.get('allow_evidence_overflow') or set(mapping) != mandatory):
            raise ValueError('Only mandatory evidence/boundary pages may exceed the target page count.')
        if not mandatory.issubset(mapping):
            raise ValueError('Crop is missing mandatory evidence/boundary pages.')
        if any(''.join(page.split()) != ''.join(originals[number - 1].split())
               for page, number in zip(pages, mapping)):
            raise ValueError('Crop original-page mapping mismatch.')
    task_hashes = {p.relative_to(root).as_posix(): digest(p)
                   for dimension in ('retrieval', 'reasoning', 'management')
                   for p in (root / dimension).glob('*.json')}
    backup.mkdir(parents=True)
    originals = [x['release_pdf'] for x in report['prepared']] + metadata_names
    for name in originals:
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        if digest(target) != digest(root / name):
            raise ValueError('Backup verification failed.')
    shutil.copy2(crops / 'report.json', backup / 'crop_report_before_replacement.json')
    status = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset': str(root), 'backup': str(backup), 'status': 'backed_up',
              'replaced_files': report['prepared_files'], 'original_pdf_hashes': before,
              'original_task_hashes': task_hashes}
    status_path = backup / 'replacement_record.json'
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    try:
        for row in report['prepared']:
            name = row['release_pdf']
            if digest(root / name) != before[name]:
                raise ValueError('Source changed after backup.')
            atomic_copy(crops / name, root / name)
        # Release metadata contains no source document names or answer annotations.
        public = {'schema_version': 1, 'page_limit': 10,
                  'policy': 'evidence + first/last + interval fillers; cropped PDF only',
                  'evidence_scope': 'conservative union of source-dataset QA evidence per document',
                  'replaced_pdf_files': report['prepared_files'],
                  'unchanged_pdf_files': len(report['conflicts']) + len(report['errors']),
                  'all_release_pdfs_within_page_limit': False,
                  'remaining_pages_delivered_as_text': False,
                  'replaced': [{'path': row['release_pdf'], 'original_pages': row['original_pages'],
                                'pages': row['output_pages'], 'original_page_numbers': row['selected_original_pages'],
                                'sha256': row['output_sha256']} for row in report['prepared']],
                  'unchanged': [{'path': row['release_pdf'], 'reason': 'mandatory_pages_exceed_limit',
                                 'minimum_required_pages': row['minimum_required_pages']}
                                for row in report['conflicts']] +
                               [{'path': row['release_pdf'], 'reason': row['error_type']}
                                for row in report['errors']]}
        if prior_metadata:
            new_paths = {row['path'] for row in public['replaced']}
            if new_paths.intersection(row['path'] for row in prior_metadata['replaced']):
                raise ValueError('Refuse replacing a previously cropped PDF again.')
            public['replaced'] = prior_metadata['replaced'] + public['replaced']
            public['unchanged'] = [row for row in prior_metadata['unchanged'] if row['path'] not in new_paths]
            public['replaced_pdf_files'] = len(public['replaced'])
            public['unchanged_pdf_files'] = len(public['unchanged'])
        public.update(schema_version=2, page_limit=report['page_limit'],
                      page_limit_is_hard_cap=not report.get('allow_evidence_overflow', False),
                      mandatory_evidence_overflow_allowed=report.get('allow_evidence_overflow', False),
                      policy='evidence + first/last; fill other pages to 10 only when mandatory pages fit; allow mandatory-only overflow')
        for row in public['replaced']:
            row['mandatory_evidence_overflow'] = row['pages'] > report['page_limit']
        public['mandatory_overflow_pdf_files'] = sum(row['mandatory_evidence_overflow'] for row in public['replaced'])
        assert public['replaced_pdf_files'] + public['unchanged_pdf_files'] == expected_pdf_count
        new_metadata.write_text(json.dumps(public, indent=2) + '\n')
        readme = (backup / 'README.md').read_text().split('\n## PDF preprocessing', 1)[0]
        readme += ('\n## PDF preprocessing\n\n'
                   f"{public['replaced_pdf_files']} of the {expected_pdf_count} packaged PDF files were replaced with verified crops.\n"
                   'Each crop retains all source-annotated evidence pages and first/last pages.\n'
                   'When those mandatory pages fit within 10 pages, other pages are added up to 10.\n'
                   'When mandatory pages exceed 10, all mandatory pages are retained, with no extras.\n'
                   'Discarded pages are not provided as text. Evidence selection conservatively\n'
                   'uses the union of source-dataset QA annotations for each document.\n\n'
                   f"{public['mandatory_overflow_pdf_files']} cropped PDFs exceed the 10-page target to preserve mandatory evidence.\n"
                   f"{public['unchanged_pdf_files']} file remains original because source evidence annotations are invalid.\n"
                   'The release therefore does not impose a universal 10-page PDF limit.\n'
                   'See `PDF_PREPROCESSING.json` for per-file status and original-page mappings.\n'
                   'Task JSON, question/answer text, timestamps, media paths and file counts\n'
                   'are unchanged. This is a revised dataset input; earlier full-PDF results\n'
                   'are not directly comparable.\n')
        (root / 'README.md').write_text(readme)
        stats = json.loads((backup / 'RELEASE_STATS.json').read_text())
        stats['pdf_preprocessing'] = {'replaced_pdf_files': public['replaced_pdf_files'],
                                      'unchanged_pdf_files': public['unchanged_pdf_files'],
                                      'cropped_pdf_page_limit': 10,
                                      'mandatory_evidence_overflow_allowed': public['mandatory_evidence_overflow_allowed'],
                                      'mandatory_overflow_pdf_files': public['mandatory_overflow_pdf_files'],
                                      'metadata': 'PDF_PREPROCESSING.json'}
        (root / 'RELEASE_STATS.json').write_text(json.dumps(stats, indent=2) + '\n')
        for name in [x['release_pdf'] for x in report['prepared']] + metadata_names[:2] + [new_metadata.name]:
            checksums[name] = digest(root / name)
        checksum_temp = backup / 'SHA256SUMS.new'
        checksum_temp.write_text(''.join(f'{checksums[name]}  {name}\n' for name in sorted(checksums)))
        atomic_copy(checksum_temp, root / 'SHA256SUMS')
        for row in report['prepared']:
            assert digest(root / row['release_pdf']) == row['output_sha256']
        for row in report['conflicts'] + report['errors']:
            assert digest(root / row['release_pdf']) == before[row['release_pdf']]
        assert all(digest(root / name) == sha for name, sha in task_hashes.items())
        assert len(list(root.rglob('*.pdf'))) == expected_pdf_count
        status.update(status='verified_complete', new_checksums_sha256=digest(root / 'SHA256SUMS'),
                      task_files_unchanged=True, remaining_pdfs_unchanged=True)
    except Exception:
        for name in originals:
            atomic_copy(backup / name, root / name)
        if not prior_metadata:
            new_metadata.unlink(missing_ok=True)
        status['status'] = 'rolled_back'
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        raise
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
