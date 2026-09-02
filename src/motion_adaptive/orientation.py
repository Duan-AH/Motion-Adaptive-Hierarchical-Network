

'Body-orientation evidence and reliability from COCO-17 tracks.'

from __future__ import annotations

from dataclasses import dataclass

import torch

from torch import Tensor

from .features import COCO17_JOINT_COUNT

ORIENTATION_FEATURE_DIM = 2

ORIENTATION_FEATURE_NAMES = ('orientation_score', 'orientation_reliability')

DEFAULT_ORIENTATION_SMOOTHING_WINDOW = 7

_TORSO_JOINTS = (5, 6, 11, 12)

_LEFT_SHOULDER = 5

_RIGHT_SHOULDER = 6

_LEFT_HIP = 11

_RIGHT_HIP = 12

@dataclass(frozen=True)
class OrientationBatch:
    """Per-frame orientation evidence, each with shape ``[B, T]``.

    Both tensors are finite and lie in ``[0, 1]``.  A score of ``0`` means
    side-on/narrow and ``1`` means front- or back-facing/broad.  Frames with
    zero reliability always carry the neutral score ``0.5``.
    """
    score: Tensor
    reliability: Tensor

def _validate_inputs(keypoints: Tensor, frame_ids: Tensor | None, sequence_mask: Tensor | None, *, confidence_threshold: float, smoothing_window: int, side_ratio: float, broad_ratio: float) -> tuple[Tensor, Tensor]:
    if keypoints.ndim != 4 or keypoints.shape[-2:] != (COCO17_JOINT_COUNT, 3):
        raise ValueError('keypoints must have shape [batch, time, 17, 3]')
    if not keypoints.is_floating_point():
        raise TypeError('keypoints must be floating point')
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError('confidence_threshold must be in [0, 1]')
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError('smoothing_window must be a positive odd integer')
    if not 0.0 <= side_ratio < broad_ratio:
        raise ValueError('orientation ratios must satisfy 0 <= side_ratio < broad_ratio')
    batch, time = keypoints.shape[:2]
    device = keypoints.device
    if sequence_mask is None:
        resolved_mask = torch.ones(batch, time, dtype=torch.bool, device=device)
    else:
        resolved_mask = sequence_mask.to(device=device, dtype=torch.bool)
    if resolved_mask.shape != (batch, time):
        raise ValueError('sequence_mask must have shape [batch, time]')
    if frame_ids is None:
        resolved_frames = torch.arange(time, device=device).unsqueeze(0).expand(batch, -1)
    else:
        resolved_frames = frame_ids.to(device=device, dtype=torch.long)
    if resolved_frames.shape != (batch, time):
        raise ValueError('frame_ids must have shape [batch, time]')
    return (resolved_frames, resolved_mask)

def _smooth_contiguous_runs(values: Tensor, weights: Tensor, frame_ids: Tensor, sequence_mask: Tensor, window: int) -> Tensor:
    """Reliability-weighted centered smoothing without crossing frame gaps."""
    if window == 1 or values.shape[1] == 0:
        return values
    result = values.clone()
    radius = window // 2
    batch, time = values.shape
    for batch_index in range(batch):
        present = sequence_mask[batch_index]
        starts = present.clone()
        if time > 1:
            starts[1:] &= ~present[:-1] | (frame_ids[batch_index, 1:] != frame_ids[batch_index, :-1] + 1)
        run_starts = torch.nonzero(starts, as_tuple=False).flatten().tolist()
        for start in run_starts:
            end = start + 1
            while end < time and bool(present[end]) and (int(frame_ids[batch_index, end]) == int(frame_ids[batch_index, end - 1]) + 1):
                end += 1
            run_values = values[batch_index, start:end]
            run_weights = weights[batch_index, start:end]
            run_length = end - start
            for offset in range(run_length):
                left = max(0, offset - radius)
                right = min(run_length, offset + radius + 1)
                local_weights = run_weights[left:right]
                denominator = local_weights.sum()
                if bool(denominator > 0):
                    result[batch_index, start + offset] = (run_values[left:right] * local_weights).sum() / denominator
    return result

def extract_orientation_features(keypoints: Tensor, *, frame_ids: Tensor | None=None, sequence_mask: Tensor | None=None, confidence_threshold: float=0.05, smoothing_window: int=DEFAULT_ORIENTATION_SMOOTHING_WINDOW, side_ratio: float=0.12, broad_ratio: float=0.55, neutral_score: float=0.5, eps: float=1e-06) -> OrientationBatch:
    """Return continuous orientation score and reliability for a pose track.

    Shoulder and hip widths are normalized by the distance between their
    midpoints.  The two width estimates vote jointly on orientation.  Their
    agreement, their line alignment, and the minimum of the four keypoint
    confidences determine reliability.  Scores receive a centered seven-frame
    reliability-weighted smoothing by default; actual frame gaps and padding
    split the smoothing runs.

    Reliability is intentionally *not* folded continuously into ``score``:
    V3-O uses it as a separate model-side gate.  Only exactly unreliable frames
    receive the neutral score, avoiding double attenuation.
    """
    frame_ids, sequence_mask = _validate_inputs(keypoints, frame_ids, sequence_mask, confidence_threshold=confidence_threshold, smoothing_window=smoothing_window, side_ratio=side_ratio, broad_ratio=broad_ratio)
    if not 0.0 <= neutral_score <= 1.0:
        raise ValueError('neutral_score must be in [0, 1]')
    if eps <= 0.0:
        raise ValueError('eps must be positive')
    xy = keypoints[..., :2]
    confidence = keypoints[..., 2]
    joint_valid = torch.isfinite(xy).all(dim=-1) & torch.isfinite(confidence) & (confidence >= confidence_threshold) & sequence_mask.unsqueeze(-1)
    torso_valid = joint_valid[..., _TORSO_JOINTS].all(dim=-1)
    clean_xy = torch.where(joint_valid.unsqueeze(-1), xy, torch.zeros_like(xy))
    shoulder_vector = clean_xy[..., _RIGHT_SHOULDER, :] - clean_xy[..., _LEFT_SHOULDER, :]
    hip_vector = clean_xy[..., _RIGHT_HIP, :] - clean_xy[..., _LEFT_HIP, :]
    shoulder_width = torch.linalg.vector_norm(shoulder_vector, dim=-1)
    hip_width = torch.linalg.vector_norm(hip_vector, dim=-1)
    shoulder_midpoint = (clean_xy[..., _LEFT_SHOULDER, :] + clean_xy[..., _RIGHT_SHOULDER, :]) * 0.5
    hip_midpoint = (clean_xy[..., _LEFT_HIP, :] + clean_xy[..., _RIGHT_HIP, :]) * 0.5
    torso_length = torch.linalg.vector_norm(shoulder_midpoint - hip_midpoint, dim=-1)
    geometry_valid = torso_valid & torch.isfinite(torso_length) & (torso_length > eps)
    denominator = torso_length.clamp_min(eps)
    shoulder_ratio = shoulder_width / denominator
    hip_ratio = hip_width / denominator
    ratio_span = broad_ratio - side_ratio
    shoulder_score = ((shoulder_ratio - side_ratio) / ratio_span).clamp(0.0, 1.0)
    hip_score = ((hip_ratio - side_ratio) / ratio_span).clamp(0.0, 1.0)
    raw_score = ((shoulder_score + hip_score) * 0.5).clamp(0.0, 1.0)
    score_agreement = (1.0 - (shoulder_score - hip_score).abs()).clamp(0.0, 1.0)
    width_agreement = (1.0 - (shoulder_width - hip_width).abs() / (shoulder_width + hip_width).clamp_min(eps)).clamp(0.0, 1.0)
    line_cosine = (shoulder_vector * hip_vector).sum(dim=-1) / (shoulder_width * hip_width).clamp_min(eps)
    line_alignment = ((line_cosine.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    torso_confidence = confidence[..., _TORSO_JOINTS].amin(dim=-1).clamp(0.0, 1.0)
    geometric_reliability = (score_agreement * width_agreement * line_alignment).clamp_min(0.0).pow(1.0 / 3.0)
    reliability = torch.where(geometry_valid, (torso_confidence * geometric_reliability).clamp(0.0, 1.0), torch.zeros_like(torso_confidence))
    reliability = torch.where(torch.isfinite(reliability), reliability, torch.zeros_like(reliability))
    safe_raw_score = torch.where(geometry_valid & torch.isfinite(raw_score), raw_score, torch.full_like(raw_score, neutral_score))
    smoothed_score = _smooth_contiguous_runs(safe_raw_score, reliability, frame_ids, sequence_mask, smoothing_window).clamp(0.0, 1.0)
    score = torch.where(reliability > 0, smoothed_score, torch.full_like(smoothed_score, neutral_score))
    return OrientationBatch(score=score, reliability=reliability)
