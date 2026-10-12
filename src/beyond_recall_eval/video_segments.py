"""Question-independent, stream-copy video chunks paired with real ASR text."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ffmpeg_binary():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def checked_ffmpeg(arguments):
    result = subprocess.run([ffmpeg_binary(), '-nostdin', '-y', '-v', 'error', *arguments],
                            capture_output=True)
    if result.returncode:
        raise ValueError('Video preprocessing failed; no lossy fallback. ' + result.stderr.decode(errors='replace')[:500])


def inspect_video(path):
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) > 1:
            raise ValueError('Segment adapter requires one video and at most one audio track.')
        if any(s.type not in {'audio', 'video'} for s in container.streams):
            raise ValueError('Additional media tracks require an explicit adapter.')
        stream = container.streams.video[0]
        rows = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Video frames must have source timestamps.')
            rows.append({'time': float(frame.pts * frame.time_base),
                         'duration': float(frame.duration * frame.time_base) if frame.duration else 1 / float(stream.average_rate),
                         'sha256': sha(frame.to_ndarray(format='rgb24').tobytes())})
        if not rows:
            raise ValueError('Video contains no decodable frames.')
        return {'frames': rows, 'width': stream.width, 'height': stream.height,
                'fps': float(stream.average_rate), 'has_audio': bool(container.streams.audio)}


def packet_hashes(path):
    import av
    hashes = {'video': [], 'audio': []}
    with av.open(str(path)) as container:
        for packet in container.demux():
            if packet.size and packet.stream.type in hashes:
                hashes[packet.stream.type].append(sha(bytes(packet)))
    return hashes


class APITranscriber:
    """Audio transcription endpoint; no visual descriptions or answers as prompts."""
    def __init__(self, config):
        self.config = dict(config)

    def __call__(self, wav):
        endpoint = os.environ[self.config.get('base_url_env', 'BR_ASR_BASE_URL')].rstrip('/')
        if not endpoint.endswith('/audio/transcriptions'):
            endpoint += '/audio/transcriptions'
        key = os.environ[self.config.get('api_key_env', 'BR_ASR_API_KEY')]
        boundary = 'br-' + uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
                f'{self.config["model"]}\r\n--{boundary}\r\n'
                'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                'Content-Type: audio/wav\r\n\r\n').encode() + wav + f'\r\n--{boundary}--\r\n'.encode()
        for attempt in range(int(self.config.get('max_retries', 2)) + 1):
            request = urllib.request.Request(endpoint, data=body, headers={
                'Authorization': 'Bearer ' + key,
                'Content-Type': 'multipart/form-data; boundary=' + boundary})
            try:
                with urllib.request.urlopen(request, timeout=self.config.get('timeout_seconds', 120)) as response:
                    result = json.load(response)
                if not isinstance(result.get('text'), str):
                    raise ValueError('ASR response lacks a text field; refusing placeholder.')
                # An empty, successful transcription means no recognized speech,
                # not that the original audio track was silent.
                return result['text'].strip()
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == int(self.config.get('max_retries', 2)):
                    raise RuntimeError(f'ASR HTTP {error.code}; response body omitted to protect credentials.') from None
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, TimeoutError):
                if attempt == int(self.config.get('max_retries', 2)):
                    raise RuntimeError('ASR network timeout or transport failure; no transcription substituted.') from None
                time.sleep(2 ** attempt)


def prepare_video(source, directory, *, segment_seconds=10, transcriber, asr_config):
    """Preserve compressed packets and decoded frames; ASR covers all audio samples."""
    import av
    source, directory = Path(source), Path(directory)
    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError('segment_seconds must be positive and finite.')
    original_hash = sha(source.read_bytes())
    settings = {'schema_version': 1, 'source_sha256': original_hash,
                'target_segment_seconds': segment_seconds, 'asr': dict(asr_config)}
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / 'record.json'
    if saved.exists():
        record = json.loads(saved.read_text())
        if record.get('settings') != settings:
            raise ValueError('Existing video cache has different preprocessing settings.')
        if record.get('status') == 'ok':
            for segment in record['segments']:
                if sha((directory / segment['file']).read_bytes()) != segment['sha256']:
                    raise ValueError('Cached video segment hash mismatch.')
            return record
    info = inspect_video(source)
    source_frames = info.pop('frames')
    origin = source_frames[0]['time']
    video_end = source_frames[-1]['time'] + source_frames[-1]['duration'] - origin
    clips = directory / 'clips'
    clips.mkdir(exist_ok=True)
    # Rebuild any incomplete attempt, never reuse unverified media.
    for path in clips.glob('*.mp4'):
        path.unlink()
    if video_end <= segment_seconds:
        import shutil
        shutil.copyfile(source, clips / '00000.mp4')
    else:
        checked_ffmpeg(['-i', str(source), '-map', '0', '-c', 'copy', '-f', 'segment',
                        '-segment_time', str(segment_seconds), '-reset_timestamps', '1',
                        str(clips / '%05d.mp4')])
    paths = sorted(clips.glob('*.mp4'))
    source_packets = packet_hashes(source)
    actual_packets = {'video': [], 'audio': []}
    offset, segments = 0, []
    for index, path in enumerate(paths):
        chunk_info = inspect_video(path)
        chunk_frames = chunk_info['frames']
        expected = source_frames[offset:offset + len(chunk_frames)]
        if [f['sha256'] for f in expected] != [f['sha256'] for f in chunk_frames]:
            raise ValueError('Stream-copy chunks did not preserve every decoded frame in order.')
        if any(abs((f['time'] - chunk_frames[0]['time']) - (s['time'] - expected[0]['time'])) > .002
               for f, s in zip(chunk_frames, expected)):
            raise ValueError('Stream-copy changed relative frame timing.')
        if (chunk_info['width'], chunk_info['height']) != (info['width'], info['height']):
            raise ValueError('Video resolution changed.')
        packets = packet_hashes(path)
        for kind in actual_packets:
            actual_packets[kind].extend(packets[kind])
        segments.append({'index': index, 'file': path.relative_to(directory).as_posix(),
                         'sha256': sha(path.read_bytes()), 'frames': len(chunk_frames),
                         'start_seconds': expected[0]['time'] - origin,
                         'first_frame': offset, 'last_frame': offset + len(chunk_frames) - 1})
        offset += len(chunk_frames)
    if offset != len(source_frames) or actual_packets != source_packets:
        raise ValueError('Segment coverage incomplete: video/audio packets or frames were lost.')
    raw = b''
    audio_start, sample_rate, channels = 0., 0, 0
    if info['has_audio']:
        with av.open(str(source)) as container:
            audio = container.streams.audio[0]
            sample_rate = audio.codec_context.sample_rate
            channels = audio.codec_context.channels
            first = next(container.decode(audio))
            if first.pts is None:
                raise ValueError('Audio requires timestamps for alignment.')
            audio_start = float(first.pts * first.time_base) - origin
        pcm = directory / 'audio_s16le.pcm'
        checked_ffmpeg(['-i', str(source), '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le',
                        '-ar', str(sample_rate), '-ac', str(channels), '-f', 's16le', str(pcm)])
        raw = pcm.read_bytes()
    total_samples = len(raw) // (2 * channels) if channels else 0
    end = max(video_end, audio_start + total_samples / sample_rate if sample_rate else video_end)
    covered = bytearray()
    for index, segment in enumerate(segments):
        # First/last ranges include any audio lead-in or tail beyond the visuals.
        start = min(0., audio_start) if index == 0 else segment['start_seconds']
        stop = segments[index + 1]['start_seconds'] if index + 1 < len(segments) else end
        segment.update(start_seconds=start, end_seconds=stop, has_audio=info['has_audio'])
        if sample_rate:
            lo = max(0, min(total_samples, math.floor((start - audio_start) * sample_rate + 1e-6)))
            hi = max(lo, min(total_samples, math.floor((stop - audio_start) * sample_rate + 1e-6)))
            pcm_part = raw[lo * 2 * channels:hi * 2 * channels]
            covered.extend(pcm_part)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm_part)
            wav_data = buffer.getvalue()
            wav_path = directory / f'audio_{index:05d}.wav'
            wav_path.write_bytes(wav_data)
            transcript_cache = directory / f'transcript_{index:05d}.json'
            transcript_key = {'wav_sha256': sha(wav_data), 'asr': dict(asr_config)}
            cached = json.loads(transcript_cache.read_text()) if transcript_cache.exists() else None
            if cached and cached.get('key') == transcript_key and isinstance(cached.get('text'), str):
                text = cached['text']
            else:
                text = transcriber(wav_data) if pcm_part else ''
                if not isinstance(text, str):
                    raise ValueError('Transcriber did not return text.')
                transcript_cache.write_text(json.dumps({'key': transcript_key, 'text': text}, ensure_ascii=False) + '\n')
            segment.update(transcript=text, transcript_status='ok' if text else 'no_recognized_speech',
                           audio_sample_start=lo, audio_sample_end=hi, asr_wav_sha256=sha(wav_data))
        else:
            segment.update(transcript='', transcript_status='no_audio')
    if bytes(covered) != raw:
        raise ValueError('ASR audio windows do not cover every decoded PCM sample exactly once.')
    record = {'status': 'ok', 'settings': settings, 'source': info,
              'original_frame_count': len(source_frames), 'segments': segments,
              'audit': {'stream_copy': True, 'decoded_frame_pixels_equal': True,
                        'relative_frame_timing_equal': True, 'video_audio_packets_equal': True,
                        'audio_sample_coverage_complete': True, 'audio_samples': total_samples,
                        'asr_audio_sample_rate': sample_rate, 'asr_audio_channels': channels,
                        'asr_audio_format': 'pcm_s16le; conversion for ASR only',
                        'transcript_timestamp_granularity': 'clip interval, not word-level',
                        'provider_video_sampling_verified': False,
                        'native_audio_understanding': False}}
    temporary = saved.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(saved)
    return record


class VideoSegmentStore:
    def __init__(self, config):
        self.path = Path(config['video_segment_manifest'])
        if sha(self.path.read_bytes()) != config.get('video_segment_manifest_sha256'):
            raise ValueError('Video segment manifest changed or is not pinned.')
        payload = json.loads(self.path.read_text())
        if payload.get('schema_version') != 1:
            raise ValueError('Unsupported video segment cache schema.')
        self.items = payload['items']
        self.root = self.path.parent

    def require(self, source_sha256):
        item = self.items.get(source_sha256)
        if not item or item.get('status') != 'ok':
            raise ValueError('Missing complete video segments and transcription for source SHA.')
        audit = item['audit']
        if not all(audit.get(k) is True for k in ('stream_copy', 'decoded_frame_pixels_equal',
                    'relative_frame_timing_equal', 'video_audio_packets_equal', 'audio_sample_coverage_complete')):
            raise ValueError('Video segment cache lacks complete input verification.')
        directory = (self.root / item['directory']).resolve()
        if not directory.is_relative_to(self.root.resolve()):
            raise ValueError('Video cache path escapes its root.')
        for segment in item['segments']:
            path = (directory / segment['file']).resolve()
            if not path.is_relative_to(directory) or sha(path.read_bytes()) != segment['sha256']:
                raise ValueError('Video clip missing or hash mismatch.')
            if segment['transcript_status'] not in {'ok', 'no_recognized_speech', 'no_audio'}:
                raise ValueError('Video clip transcription is incomplete.')
        return directory, item
