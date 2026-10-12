"""Explicit complete frames and lossless image tiles at the API model boundary."""
import hashlib
import io

from .media import ContentPart, MediaPreparer, lossless_tiles, text_part


def complete_visual_parts(parts, config):
    mode = config.get('visual_transport', 'native')
    if mode not in {'native', 'all_frames_lossless_tiles'}:
        raise ValueError('Unsupported API visual transport mode.')
    if mode == 'native':
        return parts, {'mode': mode, 'provider_sampling_bypassed': False,
                       'provider_pixels_verified': False}
    size = int(config.get('visual_tile_size', 896))
    factor = int(config.get('visual_padding_factor', 224))
    minimum = int(config.get('visual_minimum_side', 224))
    if minimum <= 0 or minimum % factor or minimum > size:
        raise ValueError('Visual minimum side must align to padding and fit the tile.')
    expanded, videos, tiles = [], [], []
    for part in parts:
        if part.kind == 'video':
            frames, audit = MediaPreparer().video_parts(part.data, part.asset_id, part.video_transcript_status)
            layout = config.get('visual_video_layout', 'individual_frames')
            if layout not in {'individual_frames', 'lossless_contact_sheets'}:
                raise ValueError('Unsupported complete-frame layout.')
            if layout == 'lossless_contact_sheets':
                frames, positions = frame_sheets(frames, size)
                audit.update(frame_layout=layout, frame_positions=positions)
            else:
                audit.update(frame_layout=layout)
            expanded.extend(frames)
            videos.append({'asset_id': part.asset_id, **audit})
        else:
            expanded.append(part)
    output = []
    for part in expanded:
        if part.kind != 'image':
            output.append(part)
            continue
        # Preserve the complete decoded RGB raster; output is PNG, never JPEG.
        for index, (image, rectangle, dimensions) in enumerate(lossless_tiles(part.data, size, factor)):
            if image.width < minimum or image.height < minimum:
                from PIL import Image
                padded = Image.new('RGB', (max(image.width, minimum), max(image.height, minimum)), 'white')
                padded.paste(image, (0, 0))
                image = padded
            buffer = io.BytesIO()
            image.save(buffer, 'PNG')
            data = buffer.getvalue()
            label = f'{part.label} tile:{index}'.strip()
            record = {'asset_id': part.asset_id, 'source_label': part.label, 'tile': index,
                      'original_size': list(dimensions), 'xywh': list(rectangle),
                      'padded_size': list(image.size), 'sha256': hashlib.sha256(data).hexdigest()}
            tiles.append(record)
            output.append(text_part(f'Asset {part.asset_id} {label}: original_size={dimensions}, '
                                    f'tile_xywh={rectangle}; white padding is outside source pixels. '
                                    'Tiles jointly contain the complete image; interpret their coordinates together.'))
            output.append(ContentPart('image', data=data, mime='image/png', asset_id=part.asset_id, label=label))
    return output, {'mode': mode, 'video_delivery': videos, 'image_tiles': tiles,
                    'source_pixels_partitioned_without_interpolation': True,
                    'provider_video_sampling_bypassed': True,
                    'provider_pixels_verified': False,
                    'scope': 'all decoded RGB pixels and frames submitted; remote processor internals are not exposed'}


def frame_sheets(parts, maximum_side):
    """Pack complete small frames at 1:1 pixels; large frames stay individual."""
    from PIL import Image
    images = [p for p in parts if p.kind == 'image']
    labels = [p.text for p in parts if p.kind == 'text']
    output, positions = [], []
    offset = 0
    while offset < len(images):
        source = images[offset]
        with Image.open(io.BytesIO(source.data)) as image:
            width, height = image.size
        columns, rows = maximum_side // width, maximum_side // height
        if not columns or not rows:
            output.extend([text_part(labels[offset]), source])
            positions.append({'frame': offset, 'source_label': source.label,
                              'xywh': [0,0,width,height], 'source_rgb_sha256': rgb_sha(source.data)})
            offset += 1
            continue
        count = min(columns*rows, len(images)-offset)
        used_columns = min(columns, count)
        used_rows = (count+columns-1)//columns
        sheet = Image.new('RGB', (used_columns*width, used_rows*height), 'white')
        label = f'frame-sheet:{offset}'
        mapping = []
        for j in range(count):
            frame = images[offset+j]
            with Image.open(io.BytesIO(frame.data)) as image:
                if image.size != (width,height) or image.mode != 'RGB':
                    raise ValueError('Video frame dimensions/color mode changed during sheet packing.')
                x, y = (j%columns)*width, (j//columns)*height
                sheet.paste(image, (x,y))
                positions.append({'frame': offset+j, 'source_label': label,
                    'xywh': [x,y,width,height], 'source_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
                mapping.append(f'{labels[offset+j]}; sheet_xywh={(x,y,width,height)}')
        buffer = io.BytesIO()
        sheet.save(buffer, 'PNG')
        output.extend([text_part('Complete chronological frames at original pixel size:\n'+'\n'.join(mapping)),
                       ContentPart('image', data=buffer.getvalue(), mime='image/png',
                                   asset_id=source.asset_id, label=label)])
        offset += count
    return output, positions


def rgb_sha(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        return hashlib.sha256(image.tobytes()).hexdigest()
