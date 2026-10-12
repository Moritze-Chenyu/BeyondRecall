from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .manifest import hash_json, model_revisions
from .models import (
    MediaAsset,
    MediaInput,
    PreparedQuestion,
    PreparedSession,
    PreparedTurn,
    Question,
    Session,
)


class CaptionStore:
    def __init__(self, path: str | Path | None):
        self.items: dict[str, str] = {}
        self.source_hashes: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        if path is None:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Caption cache must be a JSON object.")
        self.metadata = dict(payload.get("metadata", {}))
        required_metadata = {
            "schema_version",
            "generator_model",
            "generator_revision",
            "prompt_fingerprint",
            "complete",
        }
        missing = required_metadata - set(self.metadata)
        if missing:
            raise ValueError(
                f"Caption cache metadata is missing fields: {sorted(missing)}"
            )
        if self.metadata["complete"] is not True:
            raise ValueError("Formal caption cache must declare complete=true.")
        raw_items = payload.get("items", {})
        if not isinstance(raw_items, dict):
            raise ValueError("Caption cache items must be an object.")
        for media_path, value in raw_items.items():
            if not isinstance(value, dict):
                raise ValueError(f"Caption item must be an object: {media_path}")
            if value.get("status") != "ok":
                continue
            caption = value.get("caption")
            if isinstance(caption, str) and caption.strip():
                self.items[str(media_path).replace("\\", "/")] = caption.strip()
                if value.get('source_sha256'):
                    self.source_hashes[str(media_path).replace('\\', '/')] = value['source_sha256']

    def require(self, asset: MediaAsset) -> str:
        caption = self.items.get(asset.relative_path.replace("\\", "/"))
        if not caption:
            raise KeyError(f"Missing frozen caption for media {asset.asset_id}")
        expected = self.source_hashes.get(asset.relative_path.replace('\\', '/'))
        if expected:
            from .video_segments import sha
            if sha(asset.local_path.read_bytes()) != expected:
                raise ValueError('Frozen caption source hash does not match current media.')
        return caption


def caption_prompt_fingerprint(config: Mapping[str, Any]) -> str:
    prompts = config.get("prompts", {})
    return hash_json(
        {
            "image": prompts.get("image_caption"),
            "video": prompts.get("video_caption"),
            "pdf": prompts.get("pdf_caption"),
        }
    )


def validate_caption_store(
    path: str | Path,
    config: Mapping[str, Any],
) -> CaptionStore:
    store = CaptionStore(path)
    if config.get('media_processing', {}).get('quality', {}).get('caption_policy') == 'legacy_frozen':
        if store.metadata.get('representation') != 'legacy_frozen_captions':
            raise ValueError('Legacy caption profiles require an explicitly imported legacy cache.')
        if len(store.source_hashes) != len(store.items):
            raise ValueError('Every legacy caption must be bound to an exact source SHA.')
        return store
    expected = {
        "schema_version": 1,
        "generator_model": config["models"]["caption"]["model"],
        "generator_revision": model_revisions(config)["caption"],
        "prompt_fingerprint": caption_prompt_fingerprint(config),
        "complete": True,
    }
    mismatches = {
        key: {"expected": value, "actual": store.metadata.get(key)}
        for key, value in expected.items()
        if store.metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Caption cache metadata mismatch: {mismatches}")
    return store


def opaque_media(asset: MediaAsset, staging_directory: str | Path) -> MediaInput:
    staging = Path(staging_directory)
    staging.mkdir(parents=True, exist_ok=True)
    staged_path = staging / f"{asset.asset_id}{asset.local_path.suffix.lower()}"
    if not staged_path.exists():
        temporary = staged_path.with_suffix(staged_path.suffix + ".tmp")
        try:
            temporary.hardlink_to(asset.local_path)
        except OSError:
            shutil.copyfile(asset.local_path, temporary)
        temporary.replace(staged_path)
    return MediaInput(
        asset_id=asset.asset_id,
        media_type=asset.media_type,
        suffix=asset.local_path.suffix.lower(),
        _source_path=staged_path,
    )


class InputBuilder:
    def __init__(
        self,
        mode: str,
        caption_store: CaptionStore | None = None,
        caption_error: str = "fail",
        media_staging_directory: str | Path | None = None,
        media_quality: Mapping[str, Any] | None = None,
    ):
        if mode not in {"caption", "native", "text_only"}:
            raise ValueError(f"Unsupported input mode: {mode}")
        if caption_error not in {"fail", "omit"}:
            raise ValueError(f"Unsupported caption error policy: {caption_error}")
        self.mode = mode
        self.caption_store = caption_store or CaptionStore(None)
        self.caption_error = caption_error
        self.media_staging_directory = (
            Path(media_staging_directory)
            if media_staging_directory is not None
            else None
        )
        if mode == "native" and self.media_staging_directory is None:
            raise ValueError("Native mode requires an opaque media staging directory.")
        self.seen_assets: set[str] = set()
        self.media_quality = dict(media_quality or {})
        self.pdf_text_cache: dict[str, str] = {}
        self.pdf_split_cache: dict[str, tuple[MediaInput, str]] = {}
        self.pdf_split_audit: dict[str, Any] = {}
        self.video_store = None
        self.video_cache = {}
        self.video_split_audit = {}
        if mode == 'native' and self.media_quality.get('video_mode') == 'segmented_native_transcript':
            from .video_segments import VideoSegmentStore
            self.video_store = VideoSegmentStore(self.media_quality)

    def _video_fragments(self, media):
        from .dataset import opaque_id
        from .video_segments import sha
        result = []
        for asset in media:
            if asset.media_type != 'video' or self.video_store is None:
                continue
            if asset.asset_id not in self.video_cache:
                digest = sha(asset.read_bytes())
                directory, item = self.video_store.require(digest)
                chunks = []
                for segment in item['segments']:
                    clip_id = opaque_id('video_clip', f'{asset.asset_id}/{digest}/{segment["index"]}')
                    staged = Path(self.media_staging_directory) / (clip_id + '.mp4')
                    shutil.copyfile(directory / segment['file'], staged)
                    handle = MediaInput(clip_id, 'video', '.mp4', staged,
                              video_source_asset_id=asset.asset_id, video_source_sha256=digest,
                              video_segment_index=segment['index'],
                              video_start_seconds=segment['start_seconds'], video_end_seconds=segment['end_seconds'],
                              video_frame_count=segment['frames'], video_transcript_status=segment['transcript_status'])
                    text = (f'Video {asset.asset_id} clip {segment["index"]}: original timeline '
                            f'{segment["start_seconds"]:.6f}–{segment["end_seconds"]:.6f} seconds. '
                            'The attached video and this transcript refer to the same interval.\n'
                            f'Audio transcription ({segment["transcript_status"]}; automatic ASR, may contain errors):\n'
                            + (segment['transcript'] or '[No speech recognized; this does not imply silence.]'
                               if segment['has_audio'] else '[No audio track.]'))
                    chunks.append((handle, text, asset.asset_id))
                self.video_cache[asset.asset_id] = chunks
                self.video_split_audit[asset.asset_id] = {'source_sha256': digest, 'segments': len(chunks), **item['audit']}
                # The framework only receives clips and their corresponding text.
                asset._source_path.unlink(missing_ok=True)
            result.extend(self.video_cache[asset.asset_id])
            asset._source_path.unlink(missing_ok=True)
        return result

    def fragment_session(self, session):
        """Make each video/transcript pair one atomic ingest/retrieval unit."""
        if self.video_store is None or not any(a.video_segment_index is not None for a in session.media):
            return [session]
        groups, ordinary = [], []
        for turn in session.turns:
            if any(a.video_segment_index is not None for a in turn.media):
                if ordinary:
                    groups.append(tuple(ordinary))
                    ordinary = []
                groups.append((turn,))
            else:
                ordinary.append(turn)
        if ordinary:
            groups.append(tuple(ordinary))
        return [PreparedSession(session.index, session.date, group,
                    '\n'.join([f'[{session.date}]', *(f'{t.person}: {t.content}' for t in group)]),
                    tuple(a for t in group for a in t.media),
                    tuple(dict.fromkeys([session.source_units[0], *(u for t in group for u in t.source_units)])))
                for group in groups]

    def _native_pdf_text(self, content: str, media: tuple[MediaInput, ...]):
        if self.media_quality.get('pdf_mode') in {'core_pdf_remaining_text', 'fixed_cropped_pdf'}:
            from .pdf_hybrid import split_for_memory
            handles, documents = [], []
            for asset in media:
                if asset.media_type == 'document' and asset.suffix.lower() == '.pdf':
                    original = asset._source_path
                    if asset.asset_id not in self.pdf_split_cache:
                        if self.media_quality.get('pdf_mode') == 'fixed_cropped_pdf':
                            from .pdf_evidence import crop_for_memory
                            handle, audit = crop_for_memory(asset.read_bytes(), asset.asset_id, self.media_quality, self.media_staging_directory)
                            remaining = []
                        else:
                            handle, remaining, audit = split_for_memory(asset.read_bytes(), asset.asset_id,
                                                                        self.media_quality, self.media_staging_directory)
                        text = '\n\n'.join(f'{asset.asset_id} remaining PDF text original page {row["page"]}/{audit["original_page_count"]}\n{row["text"]}' for row in remaining)
                        self.pdf_split_cache[asset.asset_id] = (handle, text)
                        self.pdf_split_audit[asset.asset_id] = audit
                    asset, text = self.pdf_split_cache[asset.asset_id]
                    documents.append(text)
                    # Retain only the derived core attachment in framework staging.
                    if original != asset._source_path:
                        original.unlink(missing_ok=True)
                handles.append(asset)
            return content + ('\n\n' + '\n\n'.join(documents) if any(documents) else ''), tuple(handles)
        if self.media_quality.get('pdf_mode') != 'core_visual_full_text' or 'pdf' in self.media_quality.get('native_file_types', []):
            return content, media
        from .pdf_hybrid import extract_memory_pages, page_text_parts, select_core_pages
        handles, documents = [], []
        for asset in media:
            if asset.media_type == 'document' and asset.suffix.lower() == '.pdf':
                if asset.asset_id not in self.pdf_text_cache:
                    texts, _, _ = extract_memory_pages(asset.read_bytes(), self.media_quality)
                    # Fail before a framework ingests an incomplete scanned PDF.
                    select_core_pages(texts, int(self.media_quality.get('pdf_visual_pages', 10)))
                    self.pdf_text_cache[asset.asset_id] = '\n\n'.join(p.text for p in page_text_parts(texts, asset.asset_id))
                documents.append(self.pdf_text_cache[asset.asset_id])
                asset = replace(asset, text_in_history=True)
            handles.append(asset)
        return content + ('\n\n' + '\n\n'.join(documents) if documents else ''), tuple(handles)

    def _new_assets(self, media: tuple[MediaAsset, ...]) -> tuple[MediaAsset, ...]:
        result = []
        for asset in media:
            if asset.asset_id in self.seen_assets:
                continue
            self.seen_assets.add(asset.asset_id)
            result.append(asset)
        return tuple(result)

    def _captioned_content(
        self,
        content: str,
        media: tuple[MediaAsset, ...],
    ) -> str:
        if self.media_quality.get("pdf_mode") == "fixed_cropped_pdf" and any(
                asset.media_type == "document" and asset.local_path.suffix.lower() == ".pdf" for asset in media):
            raise ValueError("Fixed cropped PDFs require native input; original-document caption caches cannot be reused.")
        descriptions = []
        for index, asset in enumerate(media, start=1):
            try:
                caption = self.caption_store.require(asset)
            except KeyError:
                if self.caption_error == "omit":
                    continue
                raise
            descriptions.append(f"Media {index}: {caption}")
        return (
            f"{content} [{' | '.join(descriptions)}]".strip()
            if descriptions
            else content
        )

    def prepare_session(self, session: Session) -> PreparedSession:
        turns = []
        lines = [f"[{session.date}]"]
        session_media: list[MediaInput] = []
        source_units = [session.unit_id]
        for turn in session.turns:
            # Preserve media association at every occurrence: retrieval can
            # select a later reference without selecting the first occurrence.
            new_assets = turn.media
            if self.mode == "caption":
                content = self._captioned_content(turn.content, new_assets)
                prepared_media: tuple[MediaInput, ...] = ()
                represented_media_units = [asset.asset_id for asset in new_assets]
            elif self.mode == "native":
                content = turn.content
                prepared_media = tuple(
                    opaque_media(asset, self.media_staging_directory)
                    for asset in new_assets
                )
                content, prepared_media = self._native_pdf_text(content, prepared_media)
                represented_media_units = [asset.asset_id for asset in new_assets]
            else:
                content = turn.content
                prepared_media = ()
                represented_media_units = []

            turn_units = [turn.unit_id, *represented_media_units]
            turns.append(
                PreparedTurn(
                    index=turn.index,
                    person=turn.person,
                    content=content,
                    media=prepared_media,
                    source_units=tuple(turn_units),
                    timestamp=turn.timestamp,
                )
            )
            lines.append(f"{turn.person}" + (f" [{turn.timestamp}]" if turn.timestamp else "") + f": {content}")
            session_media.extend(prepared_media)
            source_units.extend(turn_units)
        if self.video_store is not None:
            expanded = []
            for turn in turns:
                chunks = self._video_fragments(turn.media)
                if not chunks:
                    expanded.append(turn)
                    continue
                other_media = tuple(a for a in turn.media if a.media_type != 'video')
                if turn.content or other_media:
                    expanded.append(replace(turn, media=other_media,
                                            source_units=(turn.source_units[0], *(a.asset_id for a in other_media))))
                for handle, transcript, source_id in chunks:
                    expanded.append(replace(turn, content=transcript, media=(handle,),
                                            source_units=(turn.source_units[0], source_id)))
            turns = expanded
            session_media = [a for t in turns for a in t.media]
            source_units = [session.unit_id, *(u for t in turns for u in t.source_units)]
            lines = [f'[{session.date}]', *(f'{t.person}' + (f' [{t.timestamp}]' if t.timestamp else '') + f': {t.content}' for t in turns)]
        return PreparedSession(
            index=session.index,
            date=session.date,
            turns=tuple(turns),
            text="\n".join(lines),
            media=tuple(session_media),
            source_units=tuple(dict.fromkeys(source_units)),
        )

    def prepare_question(self, question: Question) -> PreparedQuestion:
        # Query media is always retained, even when it also appeared in history.
        if self.mode == "caption":
            text = self._captioned_content(question.text, question.media)
            media: tuple[MediaInput, ...] = ()
        elif self.mode == "native":
            text = question.text
            media = tuple(
                opaque_media(asset, self.media_staging_directory)
                for asset in question.media
            )
            text, media = self._native_pdf_text(text, media)
            if self.video_store is not None:
                chunks = self._video_fragments(media)
                media = tuple(a for a in media if a.media_type != 'video') + tuple(c[0] for c in chunks)
                text += '\n\n' + '\n\n'.join(c[1] for c in chunks) if chunks else ''
        else:
            text = question.text
            media = ()
        represented_units = (
            tuple(asset.asset_id for asset in question.media)
            if self.mode in {"caption", "native"}
            else ()
        )
        return PreparedQuestion(
            index=question.index,
            text=text,
            media=media,
            source_units=represented_units,
            timestamp=question.timestamp,
        )


def profile_input_builder(
    profile: Mapping[str, Any],
    caption_store_path: str | Path | None,
    media_staging_directory: str | Path | None = None,
) -> InputBuilder:
    return InputBuilder(
        mode=str(profile.get("input_mode", "caption")),
        caption_store=CaptionStore(caption_store_path),
        caption_error=str(profile.get("caption_error", "fail")),
        media_staging_directory=media_staging_directory,
    )
