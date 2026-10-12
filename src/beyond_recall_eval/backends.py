"""API and local Transformers backends consume the same complete content parts."""
from __future__ import annotations
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

from .media import CapacityError, ContentPart, MediaPreparer, UnsupportedMediaError, lossless_tiles


class OpenAIAPIBackend:
    def __init__(self, config):
        self.config = dict(config)
        self.model = config['model']
        self.base_url = os.getenv(config.get('base_url_env', 'BR_API_BASE_URL'), '').rstrip('/')
        if not self.base_url.startswith(('http://', 'https://')):
            raise ValueError('Set the API base URL environment variable.')
        self.key = os.getenv(config.get('api_key_env', 'BR_API_KEY'), '')
        if config.get('api_key_required', True) and not self.key:
            raise ValueError('Set the API key environment variable.')
        self.last_audit = {}

    def message_content(self, parts):
        content = []
        for part in parts:
            if part.kind == 'text':
                content.append({'type': 'text', 'text': part.text})
            elif part.kind == 'image':
                content.append({'type': 'image_url', 'image_url': {'url': part.data_url(), 'detail': 'high'}})
            elif part.kind == 'file':
                content.append({'type': 'file', 'file': {'filename': part.label,
                                                       'file_data': part.data_url()}})
            elif part.kind == 'video':
                content.append({'type': 'video_url', 'video_url': {'url': part.data_url()}})
            else:
                raise UnsupportedMediaError('Unsupported API content part: ' + part.kind)
        return content

    def generate(self, parts, system_prompt):
        pdf_audits = []
        if self.config.get('pdf_transport') == 'all_pages_images':
            expanded = []
            for part in parts:
                if part.kind == 'file' and part.mime == 'application/pdf':
                    pages, audit = MediaPreparer({'pdf_render_dpi': self.config.get('pdf_render_dpi', 300)}).pdf_parts(part.data, part.asset_id)
                    expanded.extend(pages)
                    pdf_audits.append({'asset_id': part.asset_id, **audit})
                else:
                    expanded.append(part)
            parts = expanded
        from .visual_transport import complete_visual_parts
        parts, visual_audit = complete_visual_parts(parts, self.config)
        payload = {'model': self.model, 'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': self.message_content(parts)}],
            'temperature': self.config.get('temperature', 0),
            'max_tokens': self.config.get('max_tokens', 1024), 'stream': False}
        payload.update(self.config.get('extra_body', {}))
        # extra_body must not replace the complete payload or scoring protocol.
        for forbidden in ('model', 'messages', 'max_tokens', 'temperature', 'stream'):
            if forbidden in self.config.get('extra_body', {}):
                raise ValueError('extra_body cannot override ' + forbidden)
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        maximum = self.config.get('maximum_request_bytes')
        if maximum is not None and len(body) > int(maximum):
            raise CapacityError('Complete request exceeds configured API byte limit; no media was dropped.')
        report = self.config.get('provider_processing_report')
        if self.config.get('require_verified_processing', False) and any(p.kind != 'text' for p in parts):
            required = {'model_revision', 'processor_revision', 'audit_artifact_sha256',
                        'no_input_truncation', 'pixel_preserving_images', 'complete_file_processing'}
            if not isinstance(report, dict) or not required.issubset(report):
                raise UnsupportedMediaError('Strict API evaluation requires a versioned provider-processing report; original upload bytes alone do not prove model-side fidelity.')
            if not all(report[k] is True for k in ('no_input_truncation', 'pixel_preserving_images', 'complete_file_processing')):
                raise UnsupportedMediaError('Provider-processing report does not establish complete, pixel-preserving input.')
            digest = str(report['audit_artifact_sha256'])
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Provider audit artifact must have a SHA256 digest.')
            if not report['model_revision'] or not report['processor_revision']:
                raise ValueError('Provider processing revisions must be specified.')
        endpoint = self.base_url if self.base_url.endswith('/chat/completions') else self.base_url + '/chat/completions'
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        retries = int(self.config.get('max_retries', 2))
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
                with urllib.request.urlopen(request, timeout=float(self.config.get('timeout_seconds', 300))) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code in {413, 422}:
                    raise CapacityError('API rejected full input (HTTP ' + str(error.code) + '); no shrinking fallback.') from None
                if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                    raise RuntimeError('API HTTP ' + str(error.code) + '; response body omitted to protect credentials.') from None
                time.sleep(min(2 ** attempt, 30))
        choice = result['choices'][0]
        if choice.get('finish_reason') in {'length', 'content_filter'}:
            raise CapacityError('API completion was incomplete: ' + choice['finish_reason'])
        answer = choice['message']['content']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('API did not return a nonempty answer.')
        self.last_audit = {'backend': 'api', 'model': self.model, 'request_bytes': len(body),
                           'pdf_files_expanded_at_model_boundary': len(pdf_audits),
                           'pdf_delivery': pdf_audits,
                           'visual_delivery': visual_audit,
                           'request_sha256': hashlib.sha256(body).hexdigest(),
                           'image_part_bytes_preserved': visual_audit['mode'] == 'native',
                           'application_rgb_pixels_preserved': visual_audit['mode'] == 'all_frames_lossless_tiles',
                           'application_truncation': False,
                           'provider_processing': self.config.get('provider_processing', 'unknown; server-side resizing is not verifiable'),
                           'provider_processing_report': report,
                           'usage': result.get('usage', {})}
        return answer.strip()


class LocalTransformersBackend:
    """Qwen2.5-VL/Qwen3-VL via AutoModelForImageTextToText, no lossy resize."""
    def __init__(self, config):
        self.config = dict(config)
        self.model = config['model']
        self.processor = self.network = None
        self.last_audit = {}

    def load(self):
        if self.network is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        revision = os.getenv(self.config.get('revision_env', ''), '') or self.config.get('revision', 'main')
        self.processor = AutoProcessor.from_pretrained(self.model, revision=revision, trust_remote_code=False)
        dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.get('dtype', 'bfloat16')]
        device = self.config.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu' and dtype != torch.float32:
            dtype = torch.float32
        self.network = AutoModelForImageTextToText.from_pretrained(
            self.model, revision=revision, torch_dtype=dtype, trust_remote_code=False).to(device).eval()
        self.device = device

    def generate(self, parts, system_prompt):
        self.load()
        import torch
        # The memory interface retains a PDF file. A vision-only local model
        # expands that file only here, after memory retrieval, at its model boundary.
        expanded, pdf_files, video_delivery = [], 0, []
        for part in parts:
            if part.kind == 'file' and part.mime == 'application/pdf':
                pages, _ = MediaPreparer({'pdf_render_dpi': self.config.get('pdf_render_dpi', 300)}).pdf_parts(part.data, part.asset_id)
                expanded.extend(pages)
                pdf_files += 1
            elif part.kind == 'video':
                frames, audit = MediaPreparer().video_parts(part.data, part.asset_id, part.video_transcript_status)
                expanded.extend(frames)
                video_delivery.append({'asset_id': part.asset_id, **audit})
            else:
                expanded.append(part)
        parts = expanded
        image_processor = self.processor.image_processor
        patch = int(getattr(image_processor, 'patch_size', 14))
        merge = int(getattr(image_processor, 'merge_size', 2))
        factor = patch * merge
        tile_size = int(self.config.get('tile_size', factor * 36))
        content, images, coordinates = [], [], []
        for part in parts:
            if part.kind == 'text':
                content.append({'type': 'text', 'text': part.text})
            elif part.kind == 'image':
                for image, rectangle, dimensions in lossless_tiles(part.data, tile_size, factor):
                    content.append({'type': 'text', 'text': f'Asset {part.asset_id} {part.label} original_size={dimensions} tile_xywh={rectangle}; padding is white.'})
                    content.append({'type': 'image', 'image': image})
                    images.append(image)
                    coordinates.append({'asset_id': part.asset_id, 'xywh': rectangle,
                                        'original_size': dimensions, 'padded_size': image.size})
            else:
                raise UnsupportedMediaError('Local vision backend requires supported complete media expansion; raw files are unsupported.')
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
                    {'role': 'user', 'content': content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=bool(self.config.get('enable_thinking', False)))
        inputs = self.processor(text=[prompt], images=images or None, return_tensors='pt',
                                padding=True, truncation=False, do_resize=False)
        if images:
            grids = inputs.get('image_grid_thw')
            if grids is None or len(grids) != len(images):
                raise ValueError('Processor did not preserve all image tiles.')
            for image, grid in zip(images, grids):
                if int(grid[1]) * patch != image.height or int(grid[2]) * patch != image.width:
                    raise ValueError('Processor resized a tile despite do_resize=False; refusing degraded inference.')
        input_tokens = int(inputs['input_ids'].shape[-1])
        maximum = self.config.get('context_window')
        if maximum is None:
            maximum = getattr(getattr(self.network.config, 'text_config', self.network.config), 'max_position_embeddings', None)
        if maximum is None:
            raise ValueError('Set context_window: model capacity could not be determined.')
        output_tokens = int(self.config.get('max_tokens', 1024))
        if input_tokens + output_tokens > int(maximum):
            raise CapacityError(f'Full input requires {input_tokens}+{output_tokens} tokens; context window is {maximum}. No truncation.')
        tensors = inputs.to(self.device)
        with torch.inference_mode():
            generated = self.network.generate(**tensors, max_new_tokens=output_tokens, do_sample=False)
        completion = generated[:, input_tokens:]
        eos = self.network.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        if completion.shape[-1] >= output_tokens and (not eos_ids or int(completion[0, -1]) not in eos_ids):
            raise CapacityError('Local completion exhausted max_tokens before EOS.')
        answer = self.processor.batch_decode(completion, skip_special_tokens=True)[0].strip()
        self.last_audit = {'backend': 'local_transformers', 'model': self.model,
                           'pdf_files_expanded_at_model_boundary': pdf_files,
                           'video_delivery': video_delivery,
                           'input_tokens': input_tokens, 'image_tiles': len(images),
                           'tile_coordinates': coordinates, 'processor_do_resize': False,
                           'processor_grid_verified': True, 'application_truncation': False}
        if not answer:
            raise ValueError('Local model returned an empty answer.')
        return answer


def create_backend(config):
    kind = config.get('backend', 'api')
    if kind == 'api':
        return OpenAIAPIBackend(config)
    if kind == 'transformers':
        return LocalTransformersBackend(config)
    raise ValueError('Unknown model backend: ' + kind)
