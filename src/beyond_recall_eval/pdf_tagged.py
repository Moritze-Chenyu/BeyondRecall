"""Crop tagged PDFs without losing ActualText or retaining discarded page objects."""
import io


def export_tagged_pdf(data, page_numbers, destination):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NullObject, NumberObject

    writer = PdfWriter()
    writer.clone_document_from_reader(PdfReader(io.BytesIO(data)))
    chosen = {n - 1 for n in page_numbers}
    pages = list(writer.pages)
    selected_ids = {pages[n].indirect_reference.idnum for n in chosen}
    parent_keys = {int(pages[n]['/StructParents']) for n in chosen if '/StructParents' in pages[n]}
    for n in chosen:
        for annotation in pages[n].get('/Annots', []):
            annotation = annotation.get_object()
            if '/StructParent' in annotation:
                parent_keys.add(int(annotation['/StructParent']))
    structure = writer.root_object['/StructTreeRoot'].get_object()
    active_elements = set()

    def prune(value, inherited_page=None):
        obj = value.get_object()
        if isinstance(obj, ArrayObject):
            retained = [result for child in obj if (result := prune(child, inherited_page)) is not None]
            return ArrayObject(retained) if retained else None
        if not isinstance(obj, DictionaryObject):
            return value if inherited_page is None or inherited_page in selected_ids else None
        own_page = obj.get('/Pg')
        page_id = own_page.idnum if hasattr(own_page, 'idnum') else inherited_page
        if '/K' in obj:
            children = prune(obj.raw_get('/K'), page_id)
            if children is None:
                return None
            obj[NameObject('/K')] = children
        elif page_id is not None and page_id not in selected_ids:
            return None
        if page_id is not None and page_id not in selected_ids:
            # A cross-page container can survive through selected descendants.
            for key in ('/Pg', '/ActualText', '/Alt'):
                obj.pop(key, None)
        if getattr(value, 'idnum', None) is not None:
            active_elements.add(value.idnum)
        return value

    children = prune(structure.raw_get('/K')) if '/K' in structure else None
    structure[NameObject('/K')] = children if children is not None else ArrayObject()
    structure.pop('/IDTree', None)  # Optional IDs can retain discarded structure elements.

    def number_entries(tree):
        tree = tree.get_object()
        values = tree.get('/Nums', [])
        yield from zip(values[::2], values[1::2])
        for child in tree.get('/Kids', []):
            yield from number_entries(child)

    entries = []
    if '/ParentTree' in structure:
        for key, value in number_entries(structure['/ParentTree']):
            if int(key) not in parent_keys:
                continue
            obj = value.get_object()
            if isinstance(obj, ArrayObject):
                value = ArrayObject([child if getattr(child, 'idnum', None) in active_elements else NullObject()
                                     for child in obj])
            elif getattr(value, 'idnum', None) not in active_elements:
                continue
            entries.extend([NumberObject(key), value])
        structure[NameObject('/ParentTree')] = DictionaryObject({NameObject('/Nums'): ArrayObject(entries)})
    for index in reversed(range(len(pages))):
        if index not in chosen:
            writer.remove_page(index, clean=True)
    # Selection is in source order in the benchmark's fixed plans.
    if page_numbers != sorted(set(page_numbers)):
        raise ValueError('Tagged PDF export requires unique pages in source order.')
    # Prune unreachable objects without deduplicating font resources. Font
    # deduplication corrupted a retained page in the real Lenovo manual.
    writer.compress_identical_objects(remove_duplicates=False, remove_unreferenced=True)
    writer.write(str(destination))
