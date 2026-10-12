"""Fixed, annotation-informed document preprocessing; never select at query time."""
from .media import CapacityError


def select_evidence_pages(page_count, evidence_pages, limit=10, allow_evidence_overflow=False):
    if type(page_count) is not int or page_count <= 0 or type(limit) is not int or limit <= 0:
        raise ValueError('Page count and page budget must be positive integers.')
    if not evidence_pages or any(type(p) is not int or not 1 <= p <= page_count for p in evidence_pages):
        raise ValueError('Evidence page numbers must be nonempty and refer to original PDF pages.')
    evidence = set(evidence_pages)
    mandatory = evidence | {1, page_count}
    if len(mandatory) > limit and not allow_evidence_overflow:
        raise CapacityError(f'{len(mandatory)} mandatory evidence/boundary pages exceed the {limit}-page budget; no evidence was dropped.')
    selected = set(mandatory)
    reasons = {str(page): (['source_annotated_evidence'] if page in evidence else []) +
               (['opening_page'] if page == 1 else []) + (['closing_page'] if page == page_count else [])
               for page in mandatory}
    # Fill the remaining slots from the largest uncovered intervals, making
    # other content span the document instead of privileging a tested question.
    while len(selected) < min(page_count, limit):
        ordered = sorted(selected)
        gaps = [(b - a, -a, (a + b) // 2) for a, b in zip(ordered, ordered[1:]) if b - a > 1]
        if not gaps:
            break
        page = max(gaps)[2]
        selected.add(page)
        reasons[str(page)] = ['other_content_interval_coverage']
    return sorted(selected), reasons


def crop_for_memory(data, asset_id, config, directory):
    import hashlib
    from pathlib import Path
    from .models import MediaInput
    from .pdf_hybrid import export_core_pdf, extract_pages
    digest = hashlib.sha256(data).hexdigest()
    pages = config.get('pdf_fixed_pages', {}).get(digest)
    if pages is None:
        invalid = config.get('pdf_crop_invalid_annotations', {}).get(digest)
        if invalid:
            raise ValueError(f'Invalid source evidence annotation: pages {invalid["invalid_page_numbers"]} exceed {invalid["original_pages"]} source pages.')
        conflict = config.get('pdf_crop_conflicts', {}).get(digest)
        if conflict:
            raise CapacityError(f'PDF needs {conflict["minimum_required_pages"]} mandatory pages; fixed crop budget is {config.get("pdf_visual_pages", 10)}. No evidence was dropped.')
        raise ValueError('PDF has no verified fixed crop plan; refuse heuristic fallback.')
    texts, _ = extract_pages(data)
    count = len(texts)
    limit = int(config.get('pdf_visual_pages', 10))
    if not pages or len(set(pages)) != len(pages) or any(type(p) is not int or not 1 <= p <= count for p in pages):
        raise ValueError('Invalid fixed crop page plan.')
    required = set(config.get('pdf_evidence_pages', {}).get(digest, [])) | {1, count}
    if not required.issubset(pages):
        raise ValueError('Fixed crop is missing evidence or opening/closing pages.')
    overflow = len(pages) > limit
    if overflow and (not config.get('pdf_allow_evidence_overflow', False) or set(pages) != required):
        raise ValueError('Over-budget crop must contain only mandatory evidence/boundary pages and explicitly allow evidence overflow.')
    destination = Path(directory) / (asset_id + '_cropped.pdf')
    export_core_pdf(data, sorted(pages), destination)
    audit = {'representation': 'fixed_evidence_cropped_pdf', 'source_sha256': digest,
             'original_page_count': count, 'core_original_pages': sorted(pages),
             'discarded_original_pages': [i for i in range(1, count + 1) if i not in pages],
             'remaining_text_pages': [], 'remaining_text_characters': 0,
             'core_pdf_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
             'selection_method': 'frozen_source_evidence_opening_closing_only' if overflow else 'frozen_source_evidence_opening_closing_other_pages',
             'selected_before_query': True, 'page_limit': limit,
             'mandatory_evidence_overflow': overflow,
             'overflow_allowed': bool(config.get('pdf_allow_evidence_overflow', False))}
    return MediaInput(asset_id, 'document', '.pdf', destination, False, 'cropped_pdf',
                      tuple(sorted(pages)), count, digest), audit
