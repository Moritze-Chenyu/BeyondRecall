"""Document-only core-page selection plus unabridged, page-labelled text.

Selection never uses questions, answers or private evidence. The result is an
explicit information-reduced representation, not lossless PDF processing.
"""
from collections import Counter
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .media import UnsupportedMediaError


def extract_pages(data):
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(stream=data, filetype='pdf') as document:
            if document.needs_pass:
                raise UnsupportedMediaError('Encrypted PDF needs decryption.')
            return [page.get_text('text') for page in document], 'pymupdf'
    with tempfile.TemporaryDirectory(prefix='br_pdf_text_') as raw:
        pdf = Path(raw) / 'document.pdf'
        pdf.write_bytes(data)
        info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True,
                              timeout=60, env={**os.environ, 'LC_ALL': 'C'}).stdout.decode('utf-8')
        match = re.search(r'^Pages:\s+(\d+)\s*$', info, re.MULTILINE)
        if not match:
            raise ValueError('PDF page count cannot be determined.')
        count = int(match[1])
        text = subprocess.run(['pdftotext', '-layout', '-enc', 'UTF-8', str(pdf), '-'],
                              check=True, capture_output=True, timeout=300).stdout.decode('utf-8')
        pages = text.split('\f')
        if pages and not pages[-1].strip() and len(pages) == count + 1:
            pages.pop()
        if len(pages) != count:
            raise ValueError('Text extraction page count mismatch; refusing missing pages.')
        return pages, 'poppler-layout'


def page_text_parts(texts, asset_id):
    from .media import text_part
    return [text_part(f'{asset_id} PDF text page {i + 1}/{len(texts)}\n{text}')
            for i, text in enumerate(texts)]


def extract_memory_pages(data, config):
    texts, extractor = extract_pages(data)
    missing = [i for i, text in enumerate(texts) if not text.strip()]
    ocr = {'ocr_pages': [], 'ocr_backend': None, 'ocr_unavailable': False}
    if not missing or not config.get('pdf_ocr_missing_text', True):
        return texts, extractor, ocr
    binary = os.getenv('BR_TESSERACT_BIN', '') or config.get('pdf_ocr_binary', 'tesseract')
    binary = shutil.which(binary)
    if binary is None:
        ocr['ocr_unavailable'] = True
        # select_core_pages will retain textless pages as images or reject the
        # entire document when they exceed the declared visual budget.
        return texts, extractor, ocr
    ocr['ocr_backend'] = subprocess.run([binary, '--version'], check=True, capture_output=True,
                                        timeout=30).stdout.decode('utf-8').splitlines()[0]
    dpi = max(300, int(config.get('pdf_render_dpi', 300)))
    language = config.get('pdf_ocr_language', 'eng')
    with tempfile.TemporaryDirectory(prefix='br_pdf_ocr_') as raw:
        pdf = Path(raw) / 'document.pdf'
        pdf.write_bytes(data)
        for index in missing:
            prefix = Path(raw) / f'page-{index + 1}'
            subprocess.run(['pdftoppm', '-f', str(index + 1), '-l', str(index + 1), '-singlefile',
                            '-png', '-r', str(dpi), str(pdf), str(prefix)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
            subprocess.run([binary, str(prefix.with_suffix('.png')), str(prefix) + '-ocr',
                            '-l', language, '--psm', str(config.get('pdf_ocr_psm', 3)), 'txt', 'tsv'],
                           check=True, capture_output=True, timeout=180,
                           env={**os.environ, 'OMP_THREAD_LIMIT': '1'})
            result = Path(str(prefix) + '-ocr.txt').read_text(encoding='utf-8')
            import csv
            rows = list(csv.DictReader(Path(str(prefix) + '-ocr.tsv').read_text(encoding='utf-8').splitlines(), delimiter='\t'))
            confidence = [float(row['conf']) for row in rows if row.get('text', '').strip() and float(row['conf']) >= 0]
            texts[index] = result
            ocr['ocr_pages'].append({'page': index + 1, 'characters': len(result),
                                     'mean_word_confidence': sum(confidence) / len(confidence) if confidence else None,
                                     'language': language, 'render_dpi': dpi,
                                     'requires_human_review': True})
    return texts, extractor, ocr


def select_core_pages(texts, limit=10):
    if limit <= 0:
        raise ValueError('pdf_visual_pages must be positive.')
    n = len(texts)
    if not n:
        raise ValueError('PDF has no pages.')
    empty = [i for i, text in enumerate(texts) if not text.strip()]
    # Textless pages cannot be represented by extracted text. Keep them visually
    # or explicitly require OCR, rather than claiming they were converted.
    if len(empty) > limit and n > limit:
        raise UnsupportedMediaError('More textless pages than visual budget; OCR is required before hybrid PDF processing.')
    if n <= limit:
        return list(range(n)), {str(i + 1): ['short_document_all_pages'] for i in range(n)}
    stop = set('the and for with this that from are was were have has not but you your can will into about which their page'.split())
    counters = [Counter(t for t in re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]', text.lower()) if t not in stop)
                for text in texts]
    df = Counter(term for counter in counters for term in counter)
    weights = {term: math.log1p(freq) for term, freq in df.items() if freq > 1}
    scored = []
    reasons = {}
    for index, (text, counter) in enumerate(zip(texts, counters)):
        centrality = sum(min(freq, 3) * weights.get(term, 0) for term, freq in counter.items()) / math.sqrt(max(sum(counter.values()), 1))
        semantic = bool(re.search(r'\b(abstract|summary|conclusions?|key findings|results|takeaways)\b|\u6458\u8981|\u603b\u7ed3|\u7ed3\u8bba', text, re.I))
        visual = bool(re.search(r'\b(fig(?:ure)?\.?|tables?|charts?|diagrams?)\b|\u56fe\u8868|\u793a\u610f\u56fe', text, re.I))
        numeric_table = sum(bool(re.search(r'\d\S*\s{2,}\S+\s{2,}', line)) for line in text.splitlines())
        score = centrality + 12 * semantic + 6 * visual + min(numeric_table, 5)
        scored.append((score, -index, index))
        reasons[str(index + 1)] = ['document_term_coverage'] + (['summary_or_results'] if semantic else []) + (['figure_or_table_text'] if visual else [])
    selected = set(empty)
    for index in empty:
        reasons[str(index + 1)] = ['textless_page_requires_visual']
    for index in (0, 1, n - 1):
        if len(selected) < limit:
            selected.add(index)
            reasons[str(index + 1)] = (['textless_page_requires_visual'] if index in empty else []) + ['document_boundary']
    # Reserve roughly one third of slots for coverage across the whole document.
    bands = max(1, limit // 3)
    for band in range(bands):
        candidates = [row for row in scored if n * band // bands <= row[2] < n * (band + 1) // bands and row[2] not in selected]
        if candidates and len(selected) < limit:
            index = max(candidates)[2]
            selected.add(index)
            reasons[str(index + 1)].append('document_section_coverage')
    for _, _, index in sorted(scored, reverse=True):
        if len(selected) >= limit:
            break
        selected.add(index)
    indices = sorted(selected)
    return indices, {str(i + 1): reasons[str(i + 1)] for i in indices}


def prepare_hybrid(data, asset_id, dpi, config, texts=None, extractor=None):
    from .media import ContentPart, text_part
    if texts is None:
        texts, extractor, ocr = extract_memory_pages(data, config)
    else:
        ocr = {}
    limit = int(config.get('pdf_visual_pages', 10))
    indices, reasons = select_core_pages(texts, limit)
    overrides = config.get('pdf_page_overrides', {}).get(asset_id)
    if overrides is not None:
        if not isinstance(overrides, list) or not overrides or any(type(i) is not int or not 1 <= i <= len(texts) for i in overrides):
            raise ValueError('PDF page overrides must be a nonempty list of valid original page numbers.')
        if len(overrides) > limit or len(set(overrides)) != len(overrides):
            raise ValueError('PDF page overrides must be unique and within the visual page budget.')
        indices = sorted(i - 1 for i in overrides)
        if any(not text.strip() and i not in indices for i, text in enumerate(texts)):
            raise UnsupportedMediaError('Manual PDF selection leaves textless pages unrepresented; OCR is required.')
        reasons = {str(i + 1): ['manual_document_only_override'] for i in indices}
    parts = page_text_parts(texts, asset_id)
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(stream=data, filetype='pdf') as document:
            for index in indices:
                parts.append(text_part(f'{asset_id} selected PDF visual page {index + 1}/{len(texts)}'))
                parts.append(ContentPart('image', data=document[index].get_pixmap(dpi=dpi, alpha=False).tobytes('png'),
                                         mime='image/png', asset_id=asset_id, label=f'page:{index + 1}'))
    else:
        with tempfile.TemporaryDirectory(prefix='br_pdf_core_') as raw:
            pdf = Path(raw) / 'document.pdf'
            pdf.write_bytes(data)
            for index in indices:
                prefix = Path(raw) / f'page-{index + 1}'
                subprocess.run(['pdftoppm', '-f', str(index + 1), '-l', str(index + 1), '-singlefile',
                                '-png', '-r', str(dpi), str(pdf), str(prefix)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
                parts.append(text_part(f'{asset_id} selected PDF visual page {index + 1}/{len(texts)}'))
                parts.append(ContentPart('image', data=prefix.with_suffix('.png').read_bytes(), mime='image/png',
                                         asset_id=asset_id, label=f'page:{index + 1}'))
    return parts, {'representation': 'core_visual_pages_and_full_text', 'page_count': len(texts),
                   'delivered_pages': len(indices), 'selected_visual_pages': [i + 1 for i in indices],
                   'text_pages': len(texts), 'text_characters': sum(map(len, texts)),
                   'empty_text_pages': [i + 1 for i, text in enumerate(texts) if not text.strip()],
                   'text_extractor': extractor, 'render_dpi': dpi,
                   **ocr,
                   'selection_method': 'document_only_core_coverage_v1', 'selection_reasons': reasons,
                   'text_truncated': False, 'visual_pages_omitted': len(texts) - len(indices),
                   'information_reduced': len(texts) > len(indices)}


def split_for_memory(data, asset_id, config, directory):
    """The memory boundary receives a core PDF and ONLY remaining-page text."""
    from .models import MediaInput
    import hashlib
    texts, extractor, ocr = extract_memory_pages(data, config)
    limit = int(config.get('pdf_visual_pages', 10))
    indices, reasons = select_core_pages(texts, limit)
    overrides = config.get('pdf_page_overrides', {}).get(asset_id)
    if overrides is not None:
        if not isinstance(overrides, list) or not overrides or any(type(i) is not int or not 1 <= i <= len(texts) for i in overrides):
            raise ValueError('Invalid original page numbers in PDF override.')
        if len(overrides) > limit or len(set(overrides)) != len(overrides):
            raise ValueError('PDF overrides exceed budget or repeat pages.')
        indices = sorted(number - 1 for number in overrides)
        if any(not text.strip() and i not in indices for i, text in enumerate(texts)):
            raise UnsupportedMediaError('Unselected textless pages require OCR.')
        reasons = {str(i + 1): ['manual_document_only_override'] for i in indices}
    selected = [i + 1 for i in indices]
    remaining = [{'page': i + 1, 'text': text} for i, text in enumerate(texts) if i not in set(indices)]
    core = Path(directory) / (asset_id + '_core.pdf')
    export_core_pdf(data, selected, core)
    audit = {'representation': 'core_pdf_and_remaining_text',
             'source_sha256': hashlib.sha256(data).hexdigest(), 'original_page_count': len(texts),
             'core_original_pages': selected, 'remaining_text_pages': [row['page'] for row in remaining],
             'remaining_text_characters': sum(len(row['text']) for row in remaining),
             'core_pdf_sha256': hashlib.sha256(core.read_bytes()).hexdigest(),
             'text_extractor': extractor, 'selection_reasons': reasons, **ocr}
    handle = MediaInput(asset_id, 'document', '.pdf', core, True, 'core_pdf', tuple(selected),
                        len(texts), audit['source_sha256'])
    return handle, remaining, audit


def export_core_pdf(data, page_numbers, destination):
    """Export original selected PDF pages, preserving their vector content."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fitz
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(stream=data, filetype='pdf') as source:
            if source.xref_get_key(source.pdf_catalog(), 'StructTreeRoot')[0] == 'xref':
                from .pdf_tagged import export_tagged_pdf
                export_tagged_pdf(data, page_numbers, destination)
                return
            # Preserve the source catalog; tagged PDFs are handled separately
            # to prune references to discarded pages and retain ActualText.
            source.select([number - 1 for number in page_numbers])
            source.save(str(destination), garbage=4, deflate=True)
    else:
        with tempfile.TemporaryDirectory(prefix='br_pdf_export_') as raw:
            directory = Path(raw)
            source = directory / 'source.pdf'
            source.write_bytes(data)
            pages = []
            for number in page_numbers:
                subprocess.run(['pdfseparate', '-f', str(number), '-l', str(number), str(source),
                                str(directory / 'page-%d.pdf')], check=True, capture_output=True, timeout=60)
                pages.append(str(directory / f'page-{number}.pdf'))
            subprocess.run(['pdfunite', *pages, str(destination.resolve())], check=True,
                           capture_output=True, timeout=60)
