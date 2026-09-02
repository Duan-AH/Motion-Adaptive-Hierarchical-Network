

'COCO-17 pose and pose-envelope features.'

from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import torch

import torch.nn.functional as F

from torch import Tensor

COCO17_JOINT_COUNT = 17

POSE_FEATURES_PER_JOINT = 5

POSE_FEATURE_DIM = COCO17_JOINT_COUNT * POSE_FEATURES_PER_JOINT

MOTION_SHAPE_FEATURE_DIM = 17

@dataclass(frozen=True)
class PersonFeatureBatch:
    """Features and reliability masks for a padded batch of tracks."""
    pose: Tensor
    motion_shape: Tensor
    pose_valid: Tensor
    motion_valid: Tensor
    joint_valid: Tensor

def _masked_quantile(values: Tensor, mask: Tensor, q: float) -> Tensor:
    if not 0.0 <= q <= 1.0:
        raise ValueError(f'q must be in [0, 1], got {q}')
    if values.shape != mask.shape:
        raise ValueError('values and mask must have the same shape')
    counts = mask.sum(dim=-1)
    ordered = values.masked_fill(~mask, torch.inf).sort(dim=-1).values
    ranks = torch.floor((counts.clamp_min(1) - 1) * q).to(torch.long)
    selected = ordered.gather(-1, ranks.unsqueeze(-1)).squeeze(-1)
    return torch.where(counts > 0, selected, torch.zeros_like(selected))

def pose_envelope(xy: Tensor, joint_valid: Tensor, *, lower_quantile: float=0.05, upper_quantile: float=0.95) -> Tensor:
    """Return ``[cx, cy, width, height]`` from reliable joints.

    ``xy`` must be ``[batch, time, joints, 2]`` and ``joint_valid`` the
    corresponding boolean joint mask. Quantiles reduce the effect of one
    misplaced extremity while retaining pose-derived body shape.
    """
    if xy.ndim != 4 or xy.shape[-1] != 2:
        raise ValueError('xy must have shape [batch, time, joints, 2]')
    if joint_valid.shape != xy.shape[:-1]:
        raise ValueError('joint_valid must have shape [batch, time, joints]')
    if lower_quantile >= upper_quantile:
        raise ValueError('lower_quantile must be smaller than upper_quantile')
    low_x = _masked_quantile(xy[..., 0], joint_valid, lower_quantile)
    high_x = _masked_quantile(xy[..., 0], joint_valid, upper_quantile)
    low_y = _masked_quantile(xy[..., 1], joint_valid, lower_quantile)
    high_y = _masked_quantile(xy[..., 1], joint_valid, upper_quantile)
    width = (high_x - low_x).clamp_min(0.0)
    height = (high_y - low_y).clamp_min(0.0)
    return torch.stack(((low_x + high_x) * 0.5, (low_y + high_y) * 0.5, width, height), dim=-1)

def _masked_moving_average(values: Tensor, valid: Tensor, kernel_size: int) -> Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('smoothing kernels must be positive odd integers')
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError('values must be [batch, time, channels] with [batch, time] mask')
    if kernel_size == 1:
        return values * valid.unsqueeze(-1)
    channels = values.shape[-1]
    data = values.transpose(1, 2)
    weights = valid.to(values.dtype).unsqueeze(1).expand(-1, channels, -1)
    kernel = torch.ones(channels, 1, kernel_size, dtype=values.dtype, device=values.device)
    numerator = F.conv1d(data * weights, kernel, padding=kernel_size // 2, groups=channels)
    denominator = F.conv1d(weights, kernel, padding=kernel_size // 2, groups=channels)
    smoothed = numerator / denominator.clamp_min(1.0)
    return smoothed.transpose(1, 2) * valid.unsqueeze(-1)

def _masked_joint_moving_average(xy: Tensor, joint_valid: Tensor, kernel_size: int) -> Tensor:
    """Smooth each joint independently without admitting invalid observations."""
    if xy.ndim != 4 or xy.shape[-1] != 2 or joint_valid.shape != xy.shape[:-1]:
        raise ValueError('xy and joint_valid must have [batch, time, joints, ...] shapes')
    batch, time, joints = xy.shape[:3]
    joint_major_xy = xy.permute(0, 2, 1, 3).reshape(batch * joints, time, 2)
    joint_major_valid = joint_valid.permute(0, 2, 1).reshape(batch * joints, time)
    smoothed = _masked_moving_average(joint_major_xy, joint_major_valid, kernel_size)
    return smoothed.reshape(batch, joints, time, 2).permute(0, 2, 1, 3)

def _first_difference(values: Tensor, valid: Tensor) -> Tensor:
    result = torch.zeros_like(values)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    result[:, 1:] = (values[:, 1:] - values[:, :-1]) * pair_valid.unsqueeze(-1)
    return result

def _image_size_tensor(image_size: Tensor | Sequence[float], *, batch: int, reference: Tensor) -> Tensor:
    size = torch.as_tensor(image_size, dtype=reference.dtype, device=reference.device)
    if size.shape == (2,):
        size = size.unsqueeze(0).expand(batch, -1)
    if size.shape != (batch, 2):
        raise ValueError('image_size must be [width, height] or [batch, 2]')
    if not torch.isfinite(size).all() or (size <= 0).any():
        raise ValueError('image_size values must be finite and positive')
    return size

def extract_person_features(keypoints: Tensor, *, image_size: Tensor | Sequence[float], sequence_mask: Tensor | None=None, confidence_threshold: float=0.05, min_visible_joints: int=5, pose_smoothing: int=3, light_smoothing: int=3, strong_smoothing: int=9, eps: float=1e-06) -> PersonFeatureBatch:
    """Construct pose and global motion/shape features.

    Args:
        keypoints: Padded ``[B, T, 17, 3]`` COCO-17 ``x, y, confidence``.
        image_size: ``[width, height]`` or one row per sequence.
        sequence_mask: Prefix-valid padded-sequence mask. If omitted, all time
            steps are considered present.

    Confidence is used to define observation masks, not as a continuous
    classifier feature.  Invalid feature positions are zeroed.
    """
    if keypoints.ndim != 4 or keypoints.shape[-2:] != (COCO17_JOINT_COUNT, 3):
        raise ValueError('keypoints must have shape [batch, time, 17, 3]')
    if not keypoints.is_floating_point():
        raise TypeError('keypoints must be a floating-point tensor')
    if not 1 <= min_visible_joints <= COCO17_JOINT_COUNT:
        raise ValueError('min_visible_joints must be between 1 and 17')
    if pose_smoothing < 1 or pose_smoothing % 2 == 0:
        raise ValueError('pose_smoothing must be a positive odd integer')
    batch, time = keypoints.shape[:2]
    if sequence_mask is None:
        sequence_mask = torch.ones(batch, time, dtype=torch.bool, device=keypoints.device)
    else:
        sequence_mask = sequence_mask.to(device=keypoints.device, dtype=torch.bool)
    if sequence_mask.shape != (batch, time):
        raise ValueError('sequence_mask must have shape [batch, time]')
    xy = keypoints[..., :2]
    confidence = keypoints[..., 2]
    joint_valid = torch.isfinite(xy).all(dim=-1) & torch.isfinite(confidence) & (confidence >= confidence_threshold) & sequence_mask.unsqueeze(-1)
    clean_xy = torch.where(joint_valid.unsqueeze(-1), xy, torch.zeros_like(xy))
    envelope = pose_envelope(clean_xy, joint_valid)
    visible_count = joint_valid.sum(dim=-1)
    geometry_valid = (envelope[..., 2] > eps) & (envelope[..., 3] > eps)
    frame_valid = sequence_mask & (visible_count >= min_visible_joints) & geometry_valid
    pose_xy = _masked_joint_moving_average(clean_xy, joint_valid, pose_smoothing)
    pose_geometry = pose_envelope(pose_xy, joint_valid)
    scale = torch.linalg.vector_norm(pose_geometry[..., 2:4], dim=-1).clamp_min(eps)
    hip_mask = joint_valid[..., (11, 12)]
    hip_xy = pose_xy[..., (11, 12), :]
    hip_count = hip_mask.sum(dim=-1, keepdim=True)
    hip_center = (hip_xy * hip_mask.unsqueeze(-1)).sum(dim=-2) / hip_count.clamp_min(1)
    envelope_center = pose_geometry[..., :2]
    root = torch.where(hip_count > 0, hip_center, envelope_center)
    local_xy = (pose_xy - root.unsqueeze(-2)) / scale[..., None, None]
    local_xy = local_xy * joint_valid.unsqueeze(-1)
    local_delta = torch.zeros_like(local_xy)
    joint_pair_valid = joint_valid[:, 1:] & joint_valid[:, :-1]
    local_delta[:, 1:] = (local_xy[:, 1:] - local_xy[:, :-1]) * joint_pair_valid.unsqueeze(-1)
    pose = torch.cat((local_xy, local_delta, joint_valid.to(keypoints.dtype).unsqueeze(-1)), dim=-1).flatten(start_dim=2)
    pose = pose * frame_valid.unsqueeze(-1)
    light = _masked_moving_average(envelope, frame_valid, light_smoothing)
    strong = _masked_moving_average(envelope, frame_valid, strong_smoothing)
    size = _image_size_tensor(image_size, batch=batch, reference=keypoints)
    image_width = size[:, 0, None]
    image_height = size[:, 1, None]

    def shape_values(box: Tensor) -> Tensor:
        width = box[..., 2].clamp_min(eps)
        height = box[..., 3].clamp_min(eps)
        return torch.stack((box[..., 0] / image_width, box[..., 1] / image_height, torch.log(width / image_width), torch.log(height / image_height), torch.log(width / height)), dim=-1)
    light_shape = shape_values(light)
    strong_shape = shape_values(strong)
    strong_scale = torch.linalg.vector_norm(strong[..., 2:4], dim=-1).clamp_min(eps)

    def motion_values(box: Tensor, shape: Tensor) -> Tensor:
        center_delta = _first_difference(box[..., :2], frame_valid)
        center_delta = center_delta / strong_scale.unsqueeze(-1)
        size_delta = _first_difference(shape[..., 2:4], frame_valid)
        return torch.cat((center_delta, size_delta), dim=-1)
    light_motion = motion_values(light, light_shape)
    strong_motion = motion_values(strong, strong_shape)
    residual = torch.cat(((light[..., :2] - strong[..., :2]) / strong_scale.unsqueeze(-1), light_shape[..., 2:4] - strong_shape[..., 2:4]), dim=-1)
    motion_shape = torch.cat((strong_shape, light_motion, strong_motion, residual), dim=-1)
    motion_shape = motion_shape * frame_valid.unsqueeze(-1)
    if pose.shape[-1] != POSE_FEATURE_DIM:
        raise RuntimeError('internal pose feature dimension mismatch')
    if motion_shape.shape[-1] != MOTION_SHAPE_FEATURE_DIM:
        raise RuntimeError('internal motion feature dimension mismatch')
    return PersonFeatureBatch(pose=pose, motion_shape=motion_shape, pose_valid=frame_valid, motion_valid=frame_valid, joint_valid=joint_valid)
