"""Label-free UBnormal inference with context halos and unique core frames."""
from __future__ import annotations

import numpy as np
import torch

from .data import collate_person_sequences, slice_sample
from .model import forward_batch


def infer_video(model, video, device, max_frames=256, halo_frames=32, batch_size=16):
    if max_frames < 1 or batch_size < 1 or not 0 <= 2 * halo_frames < max_frames:
        raise ValueError('invalid inference chunk, halo, or batch size')
    core = max_frames - 2 * halo_frames
    chunks = []
    for sample in video.samples:
        length = len(sample.pose_features)
        for start in range(0, length, core):
            stop = min(length, start + core)
            left, right = max(0, start - halo_frames), min(length, stop + halo_frames)
            chunks.append((slice_sample(sample, left, right), start - left, stop - left))
    frames = np.zeros(video.frame_count, np.float32)
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            group = chunks[start:start + batch_size]
            batch = collate_person_sequences([item[0] for item in group])
            moved = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = forward_batch(model, moved)
            probabilities = torch.sigmoid(output.logits).cpu().numpy()
            valid = output.prediction_mask.cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError('non-finite person scores')
            for index, (sample, left, right) in enumerate(group):
                for offset in range(left, right):
                    frame = int(sample.frame_ids[offset])
                    if not 0 <= frame < video.frame_count:
                        raise ValueError('person frame outside full-video metadata')
                    quality = video.quality_lookup[(video.video, sample.track_id, frame)]
                    score = float(probabilities[index, offset]) if valid[index, offset] else 0.0
                    rows.append({
                        'video': video.video, 'frame_index': frame, 'track_id': sample.track_id,
                        'score': score, 'prediction_valid': bool(valid[index, offset]),
                        'pose_valid': bool(quality.pose_valid), 'motion_valid': bool(quality.motion_valid),
                        'cut_before': bool(quality.cut_before), 'quality_severity': quality.severity,
                        'quality_reasons': list(quality.reasons),
                        'processed_bbox': getattr(quality, 'processed_bbox', None),
                        'quality_metrics': dict(getattr(quality, 'metrics', {})),
                        'near_image_edge': bool(getattr(quality, 'metrics', {}).get('near_image_edge', False)),
                    })
                    frames[frame] = max(frames[frame], score)
    rows.sort(key=lambda row: (row['frame_index'], row['track_id']))
    keys = [(row['track_id'], row['frame_index']) for row in rows]
    expected = {(sample.track_id, int(frame)) for sample in video.samples for frame in sample.frame_ids}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError('halo inference duplicates or omits person frames')
    return frames, rows
