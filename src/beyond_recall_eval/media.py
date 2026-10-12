"""Full-content media preparation; never silently resize, omit, or sample."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from .models import MediaInput


class UnsupportedMediaError(ValueError):
    pass


class CapacityError(ValueError):
    pass


@dataclass
class ContentPart:
    kind: str
    text: str = ''
    data: bytes = field(default=b'', repr=False)
    mime: str = ''
    asset_id: str = ''
    label: str = ''
    video_transcript_status: str | None = None

    def data_url(self):
        return 'data:' + self.mime + ';base64,' + base64.b64encode(self.data).decode('ascii')


def text_part(text):
    return ContentPart('text', text=text)


def image_format(data):
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    raise UnsupportedMediaError('Unrecognized image encoding.')


class MediaPreparer:
    def __init__(self, config=None):
        self.config = dict(config or {})
        self.pdf_dpi = int(self.config.get('pdf_render_dpi', 300))
        if self.pdf_dpi < 300:
            raise ValueError('Quality profile requires PDF render DPI >= 300.')
        self.native_files = set(self.config.get('native_file_types', []))
        if self.config.get('pdf_mode') == 'core_visual_full_text' and 'pdf' in self.native_files:
            raise ValueError('Hybrid PDF mode cannot also declare whole native PDF upload; remove pdf from native_file_types.')
        self.video_transport = self.config.get('video_transport', 'unsupported')
        self.spreadsheet_mode = self.config.get('spreadsheet_mode', 'native_required')
        self.audit = []
        self.cache = {}

    def prepare(self, asset: MediaInput):
        # Repeated references retain their turn associations; cache only bytes.
        if asset.asset_id in self.cache:
            return self.cache[asset.asset_id]
        data = asset.read_bytes()
        suffix = asset.suffix.lower()
        entry = {'asset_id': asset.asset_id, 'source_bytes': len(data),
                 'source_sha256': hashlib.sha256(data).hexdigest(),
                 'media_type': asset.media_type, 'downsampled': False,
                 'truncated': False}
        if asset.media_type == 'image':
            from PIL import Image
            mime = image_format(data)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                entry.update(width=image.width, height=image.height, frames=getattr(image, 'n_frames', 1))
            if entry['frames'] == 1:
                parts = [ContentPart('image', data=data, mime=mime, asset_id=asset.asset_id)]
                entry['representation'] = 'original_image_bytes'
            else:
                # JPEG-compatible MPO and animated images otherwise silently lose
                # every frame after the first. Preserve every decoded frame in PNG.
                parts, frame_details = [], []
                with Image.open(io.BytesIO(data)) as image:
                    for index in range(entry['frames']):
                        image.seek(index)
                        frame = image.copy()
                        if frame.mode == 'P':
                            frame = frame.convert('RGBA' if 'transparency' in frame.info else 'RGB')
                        if frame.mode not in {'RGB', 'RGBA', 'L', 'LA', '1', 'I;16'}:
                            raise UnsupportedMediaError('Cannot encode every frame losslessly; native multi-frame support is required.')
                        details = {'frame': index, 'duration_ms': image.info.get('duration'),
                                   'size': frame.size, 'format': image.format}
                        frame_details.append(details)
                        output = io.BytesIO()
                        options = {key: frame.info[key] for key in ('exif', 'icc_profile') if key in frame.info}
                        frame.save(output, format='PNG', **options)
                        parts.append(text_part(asset.asset_id + ' image-frame ' + json.dumps(details)))
                        parts.append(ContentPart('image', data=output.getvalue(), mime='image/png',
                                                 asset_id=asset.asset_id, label=f'image-frame:{index}'))
                entry.update(representation='all_decoded_image_frames_lossless_png',
                             delivered_frames=len(frame_details), frame_details=frame_details)
        elif asset.media_type == 'document' and suffix == '.pdf':
            if asset.pdf_representation in {'core_pdf', 'cropped_pdf'}:
                remainder = 'other original pages are text in the prepared history.' if asset.pdf_representation == 'core_pdf' else 'other original pages were removed by the fixed document preprocessing.'
                parts = [text_part(f'{asset.asset_id} PDF attachment: PDF page order corresponds to original pages {list(asset.pdf_original_pages)}; {remainder}'),
                         ContentPart('file', data=data, mime='application/pdf', asset_id=asset.asset_id,
                                     label=asset.asset_id + '_core.pdf')]
                entry.update(representation='core_pdf_file', original_page_count=asset.pdf_original_page_count,
                             core_original_pages=list(asset.pdf_original_pages),
                             original_source_sha256=asset.pdf_source_sha256, remaining_text_delivery='prepared_history_or_query' if asset.pdf_representation == 'core_pdf' else 'none')
            elif self.config.get('pdf_mode') in {'core_pdf_remaining_text', 'fixed_cropped_pdf'}:
                raise ValueError('Split PDF before framework ingestion using InputBuilder; original PDF cannot bypass the memory boundary.')
            elif 'pdf' in self.native_files:
                parts = [ContentPart('file', data=data, mime='application/pdf',
                                     asset_id=asset.asset_id, label=asset.asset_id + '.pdf')]
                entry['representation'] = 'complete_native_pdf'
            else:
                mode = self.config.get('pdf_mode', 'all_pages')
                if mode == 'core_visual_full_text':
                    from .pdf_hybrid import prepare_hybrid
                    parts, details = prepare_hybrid(data, asset.asset_id, self.pdf_dpi, self.config)
                    if asset.text_in_history:
                        parts = [part for part in parts if not (part.kind == 'text' and part.text.startswith(asset.asset_id + ' PDF text page '))]
                        details['full_text_delivery'] = 'prepared_history_or_query'
                    else:
                        details['full_text_delivery'] = 'content_parts'
                elif mode == 'all_pages':
                    parts, details = self.pdf_parts(data, asset.asset_id)
                else:
                    raise ValueError('Unknown pdf_mode: ' + str(mode))
                entry.update(details)
                entry['truncated'] = bool(details.get('information_reduced'))
        elif asset.media_type == 'spreadsheet':
            if suffix.lstrip('.') in self.native_files:
                mime = mimetypes.guess_type('asset' + suffix)[0] or 'application/octet-stream'
                parts = [ContentPart('file', data=data, mime=mime, asset_id=asset.asset_id,
                                     label=asset.asset_id + suffix)]
                entry['representation'] = 'complete_native_workbook'
            elif self.spreadsheet_mode == 'complete_xml_and_render':
                parts, details = self.spreadsheet_parts(data, suffix, asset.asset_id)
                entry.update(details)
            else:
                raise UnsupportedMediaError('Backend cannot ingest complete workbook; declare native support or complete_xml_and_render.')
        elif asset.media_type == 'video':
            if self.config.get('video_mode') == 'segmented_native_transcript':
                if asset.video_segment_index is None or asset.video_transcript_status not in {'ok', 'no_recognized_speech', 'no_audio'}:
                    raise ValueError('Pair video clips and their transcription before framework ingestion.')
                entry.update(representation='stream_copy_video_clip_with_paired_asr_text',
                             original_source_asset_id=asset.video_source_asset_id,
                             original_source_sha256=asset.video_source_sha256,
                             segment_index=asset.video_segment_index,
                             start_seconds=asset.video_start_seconds, end_seconds=asset.video_end_seconds,
                             source_frames=asset.video_frame_count,
                             transcript_status=asset.video_transcript_status,
                             native_audio_understanding=False,
                             provider_video_sampling_verified=False)
            if self.video_transport == 'native_file':
                parts = [ContentPart('file', data=data, mime=mimetypes.guess_type('asset' + suffix)[0] or 'application/octet-stream', asset_id=asset.asset_id,
                                     label=asset.asset_id + suffix)]
                entry['representation'] = 'complete_native_video_with_audio'
            elif self.video_transport == 'video_url':
                parts = [ContentPart('video', data=data, mime=mimetypes.guess_type('asset' + suffix)[0] or 'video/mp4',
                                     asset_id=asset.asset_id, video_transcript_status=asset.video_transcript_status)]
                entry.setdefault('representation', 'original_video_bytes_with_audio')
            elif self.video_transport == 'all_frames':
                parts, details = self.video_parts(data, asset.asset_id)
                entry.update(details)
            else:
                raise UnsupportedMediaError('Video backend is undeclared; refusing frame sampling or audio removal.')
        else:
            raise UnsupportedMediaError('Unsupported media type: ' + asset.media_type)
        entry['parts'] = len(parts)
        self.audit.append(entry)
        self.cache[asset.asset_id] = parts
        return parts

    def pdf_parts(self, data, asset_id):
        """All pages at 300+ DPI plus all extractable text; no page cap."""
        try:
            import fitz
        except ImportError:
            fitz = None
        budget = self.config.get('maximum_pdf_pages')
        if budget is not None:
            if int(budget) <= 0:
                raise ValueError('maximum_pdf_pages must be positive; it rejects whole files rather than clipping them.')
            if fitz is not None:
                with fitz.open(stream=data, filetype='pdf') as document:
                    count = len(document)
            else:
                import re
                with tempfile.TemporaryDirectory(prefix='br_pdf_preflight_') as raw:
                    pdf = Path(raw) / 'document.pdf'
                    pdf.write_bytes(data)
                    info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True,
                                          timeout=60, env={**os.environ, 'LC_ALL': 'C'}).stdout.decode('utf-8')
                    match = re.search(r'^Pages:\s+(\d+)\s*$', info, re.MULTILINE)
                    if not match:
                        raise ValueError('Cannot verify PDF page count for capacity preflight.')
                    count = int(match[1])
            if count > int(budget):
                raise CapacityError(f'Complete PDF has {count} pages, exceeding the processing budget {budget}; no pages were omitted.')
        parts = []
        if fitz is not None:
            with fitz.open(stream=data, filetype='pdf') as document:
                if document.needs_pass:
                    raise UnsupportedMediaError('Encrypted PDF requires decryption, not omission.')
                pages = len(document)
                for index, page in enumerate(document):
                    parts.append(text_part(f'{asset_id} page {index + 1}/{pages}\n' + page.get_text()))
                    parts.append(ContentPart('image', data=page.get_pixmap(dpi=self.pdf_dpi, alpha=False).tobytes('png'),
                                             mime='image/png', asset_id=asset_id, label=f'page:{index + 1}'))
        else:
            with tempfile.TemporaryDirectory(prefix='br_pdf_') as raw:
                directory = Path(raw)
                pdf = directory / 'document.pdf'
                pdf.write_bytes(data)
                subprocess.run(['pdftoppm', '-png', '-r', str(self.pdf_dpi), str(pdf), str(directory / 'page')],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                page_files = sorted(directory.glob('page-*.png'), key=lambda p: int(p.stem.split('-')[-1]))
                pages = len(page_files)
                texts = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], check=True,
                                       capture_output=True).stdout.decode('utf-8').split('\f')
                for index, page in enumerate(page_files):
                    parts.append(text_part(f'{asset_id} page {index + 1}/{pages}\n' + (texts[index] if index < len(texts) else '')))
                    parts.append(ContentPart('image', data=page.read_bytes(), mime='image/png',
                                             asset_id=asset_id, label=f'page:{index + 1}'))
        if not pages:
            raise ValueError('PDF contains no renderable pages.')
        return parts, {'representation': 'all_pdf_pages_and_text', 'page_count': pages,
                       'delivered_pages': pages, 'render_dpi': self.pdf_dpi}

    def spreadsheet_parts(self, data, suffix, asset_id):
        if suffix not in {'.xlsx', '.xlsm'}:
            raise UnsupportedMediaError('Expanded workbook mode currently supports XLSX/XLSM only.')
        # Complete OOXML includes hidden sheets/rows, formulas, cached values,
        # styles, merged ranges, chart definitions, comments and relationships.
        # Rendered pages additionally preserve chart appearance. This is an
        # explicit representation profile, not a claim of native XLSX inference.
        parts, component_details = self.workbook_components(data, asset_id)
        xml_count = component_details['xml_components']
        embedded_count = component_details['embedded_images']
        with tempfile.TemporaryDirectory(prefix='br_workbook_') as raw:
            directory = Path(raw)
            source = directory / ('workbook' + suffix)
            source.write_bytes(data)
            profile = directory / 'lo_profile'
            runtime = directory / 'runtime'
            runtime.mkdir(mode=0o700)
            environment = dict(os.environ, XDG_RUNTIME_DIR=str(runtime),
                               XDG_CONFIG_HOME=str(directory / 'config'),
                               XDG_CACHE_HOME=str(directory / 'cache'),
                               GSETTINGS_BACKEND='memory', SAL_USE_VCLPLUGIN='svp')
            try:
                subprocess.run(['libreoffice', '-env:UserInstallation=' + profile.as_uri(), '--headless',
                                '--nologo', '--nodefault', '--nofirststartwizard', '--norestore',
                                '--convert-to', 'pdf', '--outdir', str(directory), str(source)],
                               check=True, capture_output=True, timeout=300, env=environment, cwd=directory)
            except (OSError, subprocess.SubprocessError):
                raise UnsupportedMediaError('Full workbook rendering is unavailable; no partial XML-only fallback.') from None
            pdf = source.with_suffix('.pdf')
            if not pdf.is_file():
                raise UnsupportedMediaError('Workbook rendering failed; no text-only fallback.')
            # Excel keeps all rendered pages even when standalone PDFs use the
            # explicitly reduced core-page representation.
            visual, details = self.pdf_parts(pdf.read_bytes(), asset_id)
            parts.extend(visual)
        return parts, {'representation': 'complete_workbook_xml_embedded_images_and_render',
                       'xml_components': xml_count, 'embedded_images': embedded_count,
                       'rendered_pages': details['page_count'], 'render_dpi': self.pdf_dpi,
                       'rendering_tool': 'LibreOffice; print layout may differ from original application'}

    def workbook_components(self, data, asset_id):
        parts = []
        xml_count = embedded_count = 0
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in sorted(archive.namelist()):
                content = archive.read(name)
                if name.endswith(('.xml', '.rels', '.vml')):
                    parts.append(text_part(f'{asset_id} workbook component {name}\n' + content.decode('utf-8')))
                    xml_count += 1
                elif name.startswith('xl/media/'):
                    parts.append(ContentPart('image', data=content, mime=image_format(content), asset_id=asset_id,
                                             label='embedded:' + name))
                    embedded_count += 1
                elif not name.endswith('/'):
                    raise UnsupportedMediaError('Workbook has non-rendered binary components; use native file backend.')
        return parts, {'xml_components': xml_count, 'embedded_images': embedded_count}

    def video_parts(self, data, asset_id, audio_transcript_status=None):
        import av
        from PIL import Image
        parts = []
        with av.open(io.BytesIO(data)) as container:
            has_audio = bool(container.streams.audio)
            if has_audio and audio_transcript_status not in {'ok', 'no_recognized_speech'}:
                raise UnsupportedMediaError('All-frames vision backend cannot preserve audio; use native audiovisual backend.')
            if len(container.streams.video) != 1:
                raise UnsupportedMediaError('All-frames mode requires exactly one video stream; multiple streams need a dedicated adapter.')
            if any(stream.type not in {'video', 'audio'} for stream in container.streams):
                raise UnsupportedMediaError('Additional streams cannot be silently omitted.')
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                output = io.BytesIO()
                frame.to_image().save(output, format='PNG')
                parts.append(text_part(f'{asset_id} frame {index} time_seconds={frame.time}'))
                parts.append(ContentPart('image', data=output.getvalue(), mime='image/png', asset_id=asset_id,
                                         label=f'frame:{index}'))
        if not parts:
            raise UnsupportedMediaError('Video has no decoded frames.')
        return parts, {'representation': 'every_decoded_video_frame', 'frame_count': len(parts) // 2,
                       'sampled_frames': False, 'has_audio': has_audio,
                       'audio_delivery': 'paired_transcription' if has_audio else 'no_audio',
                       'native_audio_understanding': False}


def lossless_tiles(data: bytes, tile_size=1008, factor=28):
    """Pixel-exact partition + padding; never interpolate or discard pixels."""
    from PIL import Image, ImageOps
    if tile_size <= 0 or factor <= 0 or tile_size % factor:
        raise ValueError('Tile size must be a positive multiple of the model patch factor.')
    with Image.open(io.BytesIO(data)) as image:
        if image.mode not in {'RGB', 'L', '1', 'P'} or 'transparency' in image.info:
            raise UnsupportedMediaError('Strict RGB tiling cannot preserve this image color/alpha representation; use original-file capable backend.')
        image = ImageOps.exif_transpose(image).convert('RGB')
        for y in range(0, image.height, tile_size):
            for x in range(0, image.width, tile_size):
                tile = image.crop((x, y, min(x + tile_size, image.width), min(y + tile_size, image.height)))
                width = ((tile.width + factor - 1) // factor) * factor
                height = ((tile.height + factor - 1) // factor) * factor
                padded = Image.new('RGB', (width, height), 'white')
                padded.paste(tile, (0, 0))
                yield padded, (x, y, tile.width, tile.height), image.size
