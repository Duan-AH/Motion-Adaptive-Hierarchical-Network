"""Multi-timescale motion evidence and training-only robust normalization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from torch import Tensor
from .features import COCO17_JOINT_COUNT, pose_envelope

DEFAULT_MOTION_LAGS = (3, 9, 15)

MOTION_COMPONENT_NAMES = ('person_center_motion', 'torso_center_motion', 'log_envelope_scale_change', 'robust_local_joint_motion')

MOTION_EVIDENCE_NAMES = tuple((f'lag{lag}_{component}' for lag in DEFAULT_MOTION_LAGS for component in MOTION_COMPONENT_NAMES))

MOTION_EVIDENCE_DIM = len(MOTION_EVIDENCE_NAMES)

_TORSO_JOINTS = (5, 6, 11, 12)

@dataclass(frozen=True)
class MotionEvidenceBatch:
    """Non-negative evidence with an internal reliability mask.

    ``values`` and ``component_valid`` are ``[B, T, 12]``.  Invalid entries
    are always zero.  The validity tensor is calibration metadata only; the
    model still uses its existing per-frame ``motion_valid`` gate.
    """
    values: Tensor
    component_valid: Tensor

def _masked_quantile(values: Tensor, valid: Tensor, q: float) -> Tensor:
    """Per-frame masked quantile over the last axis."""
    count = valid.sum(dim=-1)
    ordered = values.masked_fill(~valid, torch.inf).sort(dim=-1).values
    rank = torch.floor((count.clamp_min(1) - 1) * q).long()
    selected = ordered.gather(-1, rank.unsqueeze(-1)).squeeze(-1)
    return torch.where(count > 0, selected, torch.zeros_like(selected))

def _weighted_quantile(values: Tensor, weights: Tensor, q: float) -> Tensor:
    """Scalar weighted quantile, deterministic for tied observations."""
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError('weighted quantile inputs must be equally shaped vectors')
    if values.numel() == 0 or not 0.0 <= q <= 1.0:
        raise ValueError('weighted quantile needs data and q in [0, 1]')
    order = torch.argsort(values, stable=True)
    ordered = values[order]
    ordered_weights = weights[order]
    threshold = q * ordered_weights.sum()
    index = torch.searchsorted(torch.cumsum(ordered_weights, dim=0), threshold, right=False).clamp_max(values.numel() - 1)
    return ordered[index]

def _continuous_pairs(frame_ids: Tensor, base_valid: Tensor, lag: int) -> Tensor:
    """Return endpoint mask whose full interval is contiguous and valid."""
    batch, time = frame_ids.shape
    result = torch.zeros(batch, time, dtype=torch.bool, device=frame_ids.device)
    if lag >= time:
        return result
    step_valid = (frame_ids[:, 1:] == frame_ids[:, :-1] + 1) & base_valid[:, 1:] & base_valid[:, :-1]
    bad = torch.zeros(batch, time, dtype=torch.long, device=frame_ids.device)
    bad[:, 1:] = (~step_valid).long()
    prefix = torch.cumsum(bad, dim=1)
    result[:, lag:] = prefix[:, lag:] - prefix[:, :-lag] == 0
    return result

def extract_multiscale_motion_evidence(keypoints: Tensor, *, frame_ids: Tensor | None=None, sequence_mask: Tensor | None=None, motion_valid: Tensor | None=None, confidence_threshold: float=0.05, lags: Sequence[int]=DEFAULT_MOTION_LAGS, min_visible_joints: int=5, min_torso_joints: int=2, min_local_joints: int=4, local_quantile: float=0.75, eps: float=1e-06) -> MotionEvidenceBatch:
    """Extract 3/9/15-frame motion evidence from raw COCO-17 tracks.

    Each lag emits, in order, person-center motion, torso-center motion,
    absolute log pose-envelope scale change, and robust body-internal joint
    motion.  Displacements are divided by pose-envelope size and by elapsed
    frames.  A frame gap or any false ``motion_valid`` value anywhere inside
    the interval invalidates the pair, so evidence never bridges a track gap.
    """
    if keypoints.ndim != 4 or keypoints.shape[-2:] != (COCO17_JOINT_COUNT, 3):
        raise ValueError('keypoints must have shape [batch, time, 17, 3]')
    if not keypoints.is_floating_point():
        raise TypeError('keypoints must be floating point')
    resolved_lags = tuple((int(lag) for lag in lags))
    if resolved_lags != DEFAULT_MOTION_LAGS:
        raise ValueError(f'motion_cluster_v2 requires lags {DEFAULT_MOTION_LAGS}')
    if not 1 <= min_visible_joints <= COCO17_JOINT_COUNT:
        raise ValueError('min_visible_joints must be in [1, 17]')
    if not 1 <= min_torso_joints <= len(_TORSO_JOINTS):
        raise ValueError('min_torso_joints must be in [1, 4]')
    if not 1 <= min_local_joints <= COCO17_JOINT_COUNT:
        raise ValueError('min_local_joints must be in [1, 17]')
    if not 0.5 <= local_quantile < 1.0:
        raise ValueError('local_quantile must be in [0.5, 1.0)')
    batch, time = keypoints.shape[:2]
    device = keypoints.device
    if sequence_mask is None:
        sequence_mask = torch.ones(batch, time, dtype=torch.bool, device=device)
    else:
        sequence_mask = sequence_mask.to(device=device, dtype=torch.bool)
    if motion_valid is None:
        motion_valid = sequence_mask
    else:
        motion_valid = motion_valid.to(device=device, dtype=torch.bool) & sequence_mask
    if sequence_mask.shape != (batch, time) or motion_valid.shape != (batch, time):
        raise ValueError('sequence_mask and motion_valid must have shape [batch, time]')
    if frame_ids is None:
        frame_ids = torch.arange(time, device=device).unsqueeze(0).expand(batch, -1)
    else:
        frame_ids = frame_ids.to(device=device, dtype=torch.long)
    if frame_ids.shape != (batch, time):
        raise ValueError('frame_ids must have shape [batch, time]')
    xy = keypoints[..., :2]
    confidence = keypoints[..., 2]
    joint_valid = torch.isfinite(xy).all(dim=-1) & torch.isfinite(confidence) & (confidence >= confidence_threshold) & sequence_mask.unsqueeze(-1)
    clean_xy = torch.where(joint_valid.unsqueeze(-1), xy, torch.zeros_like(xy))
    envelope = pose_envelope(clean_xy, joint_valid)
    scale = torch.linalg.vector_norm(envelope[..., 2:4], dim=-1)
    envelope_valid = (joint_valid.sum(dim=-1) >= min_visible_joints) & torch.isfinite(envelope).all(dim=-1) & (scale > eps)
    torso_mask = joint_valid[..., _TORSO_JOINTS]
    torso_count = torso_mask.sum(dim=-1)
    torso_center = (clean_xy[..., _TORSO_JOINTS, :] * torso_mask.unsqueeze(-1)).sum(dim=-2) / torso_count.clamp_min(1).unsqueeze(-1)
    torso_valid = torso_count >= min_torso_joints
    local_xy = (clean_xy - torso_center.unsqueeze(-2)) / scale.clamp_min(eps)[..., None, None]
    local_xy = torch.where(joint_valid.unsqueeze(-1), local_xy, torch.zeros_like(local_xy))
    base_valid = motion_valid & envelope_valid
    values_by_lag: list[Tensor] = []
    valid_by_lag: list[Tensor] = []
    for lag in resolved_lags:
        pair_continuous = _continuous_pairs(frame_ids, base_valid, lag)
        previous_envelope = torch.zeros_like(envelope)
        previous_envelope[:, lag:] = envelope[:, :-lag]
        previous_scale = torch.zeros_like(scale)
        previous_scale[:, lag:] = scale[:, :-lag]
        pair_scale = ((scale + previous_scale) * 0.5).clamp_min(eps)
        center_motion = torch.linalg.vector_norm(envelope[..., :2] - previous_envelope[..., :2], dim=-1) / (pair_scale * float(lag))
        previous_torso = torch.zeros_like(torso_center)
        previous_torso[:, lag:] = torso_center[:, :-lag]
        previous_torso_valid = torch.zeros_like(torso_valid)
        previous_torso_valid[:, lag:] = torso_valid[:, :-lag]
        torso_pair_valid = pair_continuous & torso_valid & previous_torso_valid
        torso_motion = torch.linalg.vector_norm(torso_center - previous_torso, dim=-1) / (pair_scale * float(lag))
        log_scale_change = (torch.log(scale.clamp_min(eps)) - torch.log(previous_scale.clamp_min(eps))).abs() / float(lag)
        previous_local = torch.zeros_like(local_xy)
        previous_local[:, lag:] = local_xy[:, :-lag]
        previous_joint_valid = torch.zeros_like(joint_valid)
        previous_joint_valid[:, lag:] = joint_valid[:, :-lag]
        common_joint = joint_valid & previous_joint_valid
        local_count = common_joint.sum(dim=-1)
        local_distance = torch.linalg.vector_norm(local_xy - previous_local, dim=-1)
        local_motion = _masked_quantile(local_distance, common_joint, local_quantile) / float(lag)
        local_pair_valid = pair_continuous & (local_count >= min_local_joints)
        lag_valid = pair_continuous & torso_pair_valid & local_pair_valid
        block = torch.stack((center_motion, torso_motion, log_scale_change, local_motion), dim=-1)
        block = torch.where(lag_valid.unsqueeze(-1) & torch.isfinite(block), block.clamp_min(0.0), torch.zeros_like(block))
        values_by_lag.append(block)
        valid_by_lag.append(lag_valid.unsqueeze(-1).expand_as(block))
    values = torch.cat(values_by_lag, dim=-1)
    component_valid = torch.cat(valid_by_lag, dim=-1)
    if values.shape[-1] != MOTION_EVIDENCE_DIM:
        raise RuntimeError('internal motion evidence dimension mismatch')
    return MotionEvidenceBatch(values=values, component_valid=component_valid)

def _balanced_frame_weights(video_ids: Sequence[str], track_ids: Sequence[str], *, reference: Tensor) -> Tensor:
    """Equal video, then equal track per video, then equal frame per track."""
    if len(video_ids) != len(track_ids):
        raise ValueError('video_ids and track_ids must have equal length')
    video_tracks: dict[str, set[str]] = {}
    track_counts: dict[tuple[str, str], int] = {}
    for video, track in zip(video_ids, track_ids):
        video = str(video)
        track = str(track)
        video_tracks.setdefault(video, set()).add(track)
        key = (video, track)
        track_counts[key] = track_counts.get(key, 0) + 1
    video_count = len(video_tracks)
    weights = [1.0 / (video_count * len(video_tracks[str(video)]) * track_counts[str(video), str(track)]) for video, track in zip(video_ids, track_ids)]
    return torch.tensor(weights, dtype=reference.dtype, device=reference.device)

def fit_motion_calibration(evidence: Tensor, *, video_ids: Sequence[str], track_ids: Sequence[str], valid: Tensor | None=None, eps: float=1e-08) -> dict[str, Any]:
    """Fit robust scales and clipping on training evidence only; no labels are accepted."""
    if evidence.ndim != 2 or evidence.shape[1] != MOTION_EVIDENCE_DIM:
        raise ValueError(f'evidence must have shape [N, {MOTION_EVIDENCE_DIM}]')
    if evidence.shape[0] != len(video_ids) or evidence.shape[0] != len(track_ids):
        raise ValueError('one video_id and track_id is required per evidence row')
    if valid is None:
        valid = torch.ones(evidence.shape[0], dtype=torch.bool, device=evidence.device)
    else:
        valid = valid.to(device=evidence.device, dtype=torch.bool)
    if valid.shape != (evidence.shape[0],):
        raise ValueError('valid must have shape [N]')
    valid = valid & torch.isfinite(evidence).all(dim=-1) & (evidence >= 0).all(dim=-1)
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if indices.numel() < 3:
        raise ValueError('at least three valid training frames are required')
    selected = evidence[indices].detach().to(dtype=torch.float64)
    selected_videos = [str(video_ids[int(index)]) for index in indices.cpu().tolist()]
    selected_tracks = [str(track_ids[int(index)]) for index in indices.cpu().tolist()]
    frame_weights = _balanced_frame_weights(selected_videos, selected_tracks, reference=selected)
    scales = torch.stack([_weighted_quantile(selected[:, column], frame_weights, 0.75) for column in range(MOTION_EVIDENCE_DIM)]).clamp_min(eps)
    normalized = torch.log1p(selected / scales)
    clip_upper = torch.stack([_weighted_quantile(normalized[:, column], frame_weights, 0.995) for column in range(MOTION_EVIDENCE_DIM)]).clamp_min(eps)
    normalized = torch.minimum(normalized, clip_upper)
    feature_weights = torch.full((MOTION_EVIDENCE_DIM,), 1.0 / MOTION_EVIDENCE_DIM, dtype=selected.dtype, device=selected.device)
    return {'robust_scales': [float(v) for v in scales.cpu().tolist()], 'clip_upper': [float(v) for v in clip_upper.cpu().tolist()], 'initial_feature_weights': [float(v) for v in feature_weights.cpu().tolist()]}
