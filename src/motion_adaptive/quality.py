

'Pose-only observation quality gates for person tracks.\n\nThis module is deliberately independent from UBnormal ground truth.  It never\nreads instance masks, official object identities, or normal/abnormal labels.\nThe same interface can therefore be called while preparing training data and\nwhile running inference on an unseen video.\n\nThe gates are conservative: an unusual pose or a large coherent translation\nis not treated as bad data.  Medium evidence disables only the affected model\nbranch.  A new track segment is requested only for an explicit pre-existing\nidentity warning, a frame gap, or very strong temporal corruption evidence.\n'

from __future__ import annotations

from dataclasses import dataclass, field

import math

from typing import Any, Mapping, Sequence

import numpy as np

COCO17_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))

COCO17_LEFT_RIGHT: tuple[tuple[int, int], ...] = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

@dataclass(frozen=True)
class QualityConfig:
    confidence_threshold: float = 0.05
    min_pose_joints: int = 8
    severe_joint_count: int = 5
    min_common_joints: int = 6
    min_common_bones: int = 5
    low_mean_confidence: float = 0.18
    edge_margin_ratio: float = 0.015
    motion_overlap_iou: float = 0.3
    pose_overlap_iou: float = 0.55
    bone_jump_threshold: float = 0.55
    severe_bone_jump: float = 1.0
    joint_jump_threshold: float = 0.55
    severe_joint_jump: float = 1.1
    swap_improvement_ratio: float = 0.55
    swap_min_direct_displacement: float = 0.08
    min_swap_pairs: int = 3
    envelope_log_scale_jump: float = 1.2
    boundary_warmup_frames: int = 2

    def validate(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must be in [0, 1]')
        if not 1 <= self.severe_joint_count <= self.min_pose_joints <= 17:
            raise ValueError('joint-count thresholds must satisfy 1 <= severe <= pose <= 17')
        if self.min_common_joints < 1 or self.min_common_bones < 1:
            raise ValueError('common joint/bone counts must be positive')
        if not 0.0 <= self.edge_margin_ratio < 0.5:
            raise ValueError('edge_margin_ratio must be in [0, 0.5)')
        if not 0.0 <= self.motion_overlap_iou <= self.pose_overlap_iou <= 1.0:
            raise ValueError('overlap thresholds are inconsistent')
        if self.boundary_warmup_frames < 0:
            raise ValueError('boundary_warmup_frames must be non-negative')

@dataclass(frozen=True)
class PoseQualityObservation:
    """One tracked pose observation; all fields are available at inference."""
    video: str
    track_id: str
    frame_index: int
    keypoints: np.ndarray
    predicted_bbox: tuple[float, float, float, float] | None = None
    existing_identity_break: bool = False
    existing_break_reason: str | None = None
    preprocess_reasons: tuple[str, ...] = ()
    preprocess_metrics: Mapping[str, float | int | bool | str | None] = field(default_factory=dict)

@dataclass(frozen=True)
class FrameQuality:
    video: str
    track_id: str
    frame_index: int
    pose_valid: bool
    motion_valid: bool
    cut_before: bool
    severity: str
    reasons: tuple[str, ...]
    unknown_reasons: tuple[str, ...]
    metrics: Mapping[str, float | int | bool | str | None]
    processed_keypoints: tuple[float, ...] = ()
    processed_bbox: tuple[float, float, float, float] | None = None

@dataclass(frozen=True)
class VideoQualityResult:
    """Quality decisions keyed by ``(video, predicted_track_id)``."""
    tracks: Mapping[tuple[str, str], tuple[FrameQuality, ...]]

    @property
    def frame_count(self) -> int:
        return sum((len(frames) for frames in self.tracks.values()))

    @property
    def frame_lookup(self) -> Mapping[tuple[str, str, int], FrameQuality]:
        return {(frame.video, frame.track_id, frame.frame_index): frame for frames in self.tracks.values() for frame in frames}

@dataclass
class _MutableQuality:
    observation: PoseQualityObservation
    keypoints: np.ndarray
    valid_joints: np.ndarray
    bbox: tuple[float, float, float, float] | None
    pose_valid: bool = True
    motion_valid: bool = True
    cut_before: bool = False
    reasons: list[str] = field(default_factory=list)
    unknown_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | bool | None] = field(default_factory=dict)

    def add_reason(self, reason: str, *, unknown: bool=False) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        if unknown and reason not in self.unknown_reasons:
            self.unknown_reasons.append(reason)

def _as_keypoints(value: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (51,):
        array = array.reshape(17, 3)
    if array.shape != (17, 3):
        raise ValueError(f'keypoints must have shape (17, 3) or (51,), got {array.shape}')
    return array

def _valid_bbox(value: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError('predicted_bbox must contain four numbers')
    box = tuple((float(item) for item in value))
    if not all((math.isfinite(item) for item in box)):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box

def _pose_bbox(keypoints: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float] | None:
    xy = keypoints[valid, :2]
    if len(xy) < 2:
        return None
    low = np.quantile(xy, 0.05, axis=0)
    high = np.quantile(xy, 0.95, axis=0)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        return None
    if high[0] <= low[0] or high[1] <= low[1]:
        return None
    return (float(low[0]), float(low[1]), float(high[0]), float(high[1]))

def _bbox_scale(box: Sequence[float] | None) -> float:
    if box is None:
        return 1.0
    return max(1.0, math.hypot(box[2] - box[0], box[3] - box[1]))

def _bbox_iou(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1, y1 = (max(left[0], right[0]), max(left[1], right[1]))
    x2, y2 = (min(left[2], right[2]), min(left[3], right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)

def _normalized_pose(item: _MutableQuality) -> np.ndarray:
    box = item.bbox
    if box is None:
        center = np.zeros(2, dtype=np.float64)
    else:
        center = np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    return (item.keypoints[:, :2] - center) / _bbox_scale(box)

def _bone_profile(item: _MutableQuality) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.zeros(len(COCO17_EDGES), dtype=np.float64)
    valid = np.zeros(len(COCO17_EDGES), dtype=bool)
    scale = _bbox_scale(item.bbox)
    for index, (start, end) in enumerate(COCO17_EDGES):
        if item.valid_joints[start] and item.valid_joints[end]:
            lengths[index] = np.linalg.norm(item.keypoints[start, :2] - item.keypoints[end, :2]) / scale
            valid[index] = lengths[index] > 1e-06
    if valid.any():
        lengths[valid] /= max(float(np.median(lengths[valid])), 1e-06)
    return (lengths, valid)

def _temporal_metrics(previous: _MutableQuality, current: _MutableQuality, config: QualityConfig) -> dict[str, float | int | bool | None]:
    previous_pose = _normalized_pose(previous)
    current_pose = _normalized_pose(current)
    common = previous.valid_joints & current.valid_joints
    joint_jump: float | None = None
    coherent_joint_motion: float | None = None
    peak_joint_jump: float | None = None
    if int(common.sum()) >= config.min_common_joints:
        displacements = np.linalg.norm(current_pose[common] - previous_pose[common], axis=1)
        coherent_joint_motion = float(np.median(displacements))
        peak_joint_jump = float(np.max(displacements))
        joint_jump = max(0.0, peak_joint_jump - coherent_joint_motion)
    previous_bones, previous_bone_valid = _bone_profile(previous)
    current_bones, current_bone_valid = _bone_profile(current)
    common_bones = previous_bone_valid & current_bone_valid
    bone_jump: float | None = None
    if int(common_bones.sum()) >= config.min_common_bones:
        ratio = np.log(np.maximum(current_bones[common_bones], 1e-06) / np.maximum(previous_bones[common_bones], 1e-06))
        bone_jump = float(np.quantile(np.abs(ratio), 0.9))
    direct_values: list[float] = []
    swapped_values: list[float] = []
    swap_pair_votes = 0
    evaluated_swap_pairs = 0
    for left, right in COCO17_LEFT_RIGHT:
        if common[left] and common[right]:
            pair_direct = float(0.5 * (np.linalg.norm(current_pose[left] - previous_pose[left]) + np.linalg.norm(current_pose[right] - previous_pose[right])))
            pair_swapped = float(0.5 * (np.linalg.norm(current_pose[left] - previous_pose[right]) + np.linalg.norm(current_pose[right] - previous_pose[left])))
            direct_values.append(pair_direct)
            swapped_values.append(pair_swapped)
            if pair_direct >= config.swap_min_direct_displacement:
                evaluated_swap_pairs += 1
                if pair_swapped <= config.swap_improvement_ratio * pair_direct:
                    swap_pair_votes += 1
    direct = float(np.median(direct_values)) if direct_values else None
    swapped = float(np.median(swapped_values)) if swapped_values else None
    swap_ratio = None if direct is None or swapped is None or direct <= 1e-06 else float(swapped / direct)
    swap_suspected = bool(evaluated_swap_pairs >= config.min_swap_pairs and swap_pair_votes >= config.min_swap_pairs)
    scale_change: float | None = None
    if previous.bbox is not None and current.bbox is not None:
        scale_change = abs(math.log(_bbox_scale(current.bbox) / _bbox_scale(previous.bbox)))
    return {'common_joint_count': int(common.sum()), 'common_bone_count': int(common_bones.sum()), 'joint_jump': joint_jump, 'coherent_joint_motion': coherent_joint_motion, 'peak_joint_jump': peak_joint_jump, 'bone_jump': bone_jump, 'direct_lr_displacement': direct, 'swapped_lr_displacement': swapped, 'left_right_swap_ratio': swap_ratio, 'left_right_swap_pairs_evaluated': evaluated_swap_pairs, 'left_right_swap_pair_votes': swap_pair_votes, 'left_right_swap_suspected': swap_suspected, 'envelope_log_scale_change': scale_change}

def _base_quality(observation: PoseQualityObservation, image_size: tuple[float, float], config: QualityConfig) -> _MutableQuality:
    keypoints = _as_keypoints(observation.keypoints)
    valid = np.isfinite(keypoints).all(axis=1) & (keypoints[:, 2] >= config.confidence_threshold)
    supplied_bbox = _valid_bbox(observation.predicted_bbox)
    bbox = supplied_bbox or _pose_bbox(keypoints, valid)
    item = _MutableQuality(observation, keypoints, valid, bbox)
    count = int(valid.sum())
    mean_confidence = float(keypoints[valid, 2].mean()) if count else 0.0
    item.metrics.update(valid_joint_count=count, mean_joint_confidence=mean_confidence, bbox_available=bbox is not None)
    item.metrics.update(observation.preprocess_metrics)
    for reason in observation.preprocess_reasons:
        item.add_reason(str(reason))
    if count < config.min_pose_joints or mean_confidence < config.low_mean_confidence:
        item.pose_valid = False
        item.motion_valid = False
        reason = 'too_few_trusted_joints' if count < config.min_pose_joints else 'low_joint_confidence'
        item.add_reason(reason, unknown=True)
    if count < config.severe_joint_count or bbox is None:
        item.pose_valid = False
        item.motion_valid = False
        item.add_reason('severely_incomplete_pose', unknown=True)
    width, height = image_size
    near_edge = False
    if bbox is not None:
        x_margin, y_margin = (width * config.edge_margin_ratio, height * config.edge_margin_ratio)
        near_edge = bbox[0] <= x_margin or bbox[1] <= y_margin or bbox[2] >= width - x_margin or (bbox[3] >= height - y_margin)
    item.metrics['near_image_edge'] = near_edge
    if near_edge:
        item.motion_valid = False
        item.add_reason('near_image_edge')
    return item

def _apply_overlap_gates(by_frame: Mapping[int, list[_MutableQuality]], config: QualityConfig) -> None:
    for frame_items in by_frame.values():
        for index, item in enumerate(frame_items):
            maximum = 0.0
            for other_index, other in enumerate(frame_items):
                if index != other_index and item.observation.track_id != other.observation.track_id:
                    maximum = max(maximum, _bbox_iou(item.bbox, other.bbox))
            item.metrics['max_other_pose_iou'] = maximum
            if maximum >= config.motion_overlap_iou:
                item.motion_valid = False
                item.add_reason('overlapping_pose_envelopes')
            if maximum >= config.pose_overlap_iou:
                item.pose_valid = False
                item.add_reason('severe_pose_overlap', unknown=True)

def _apply_temporal_gates(items: list[_MutableQuality], config: QualityConfig) -> None:
    for index, current in enumerate(items):
        if current.observation.existing_identity_break:
            current.cut_before = index > 0
            current.motion_valid = False
            current.add_reason('existing_identity_break', unknown=True)
            if current.observation.existing_break_reason:
                current.metrics['existing_break_reason'] = current.observation.existing_break_reason
        if index == 0:
            continue
        previous = items[index - 1]
        gap = current.observation.frame_index - previous.observation.frame_index
        current.metrics['frame_delta'] = gap
        if gap != 1:
            current.cut_before = True
            current.motion_valid = False
            previous.motion_valid = False
            current.add_reason('frame_gap', unknown=True)
            previous.add_reason('motion_before_frame_gap')
            continue
        metrics = _temporal_metrics(previous, current, config)
        current.metrics.update(metrics)
        bone_jump = metrics['bone_jump']
        joint_jump = metrics['joint_jump']
        scale_change = metrics['envelope_log_scale_change']
        swap = bool(metrics['left_right_swap_suspected'])
        moderate_bone = bone_jump is not None and bone_jump >= config.bone_jump_threshold
        severe_bone = bone_jump is not None and bone_jump >= config.severe_bone_jump
        moderate_joint = joint_jump is not None and joint_jump >= config.joint_jump_threshold
        severe_joint = joint_jump is not None and joint_jump >= config.severe_joint_jump
        severe_scale = scale_change is not None and scale_change >= config.envelope_log_scale_jump
        if moderate_bone:
            current.pose_valid = False
            current.add_reason('sudden_bone_length_change', unknown=True)
        if moderate_joint:
            current.pose_valid = False
            current.add_reason('incoherent_joint_jump', unknown=True)
        if swap:
            current.pose_valid = False
            current.add_reason('left_right_label_swap_suspected', unknown=True)
        if severe_scale and (moderate_bone or moderate_joint):
            current.motion_valid = False
            current.add_reason('unstable_pose_envelope', unknown=True)
        severe_corruption = severe_bone and severe_joint or (swap and severe_joint) or (severe_scale and severe_bone and moderate_joint)
        if severe_corruption:
            current.cut_before = True
            current.pose_valid = False
            current.motion_valid = False
            previous.motion_valid = False
            current.add_reason('severe_temporal_corruption_identity_break', unknown=True)

def _apply_boundary_warmup(items: list[_MutableQuality], config: QualityConfig) -> None:
    starts = [0] + [index for index, item in enumerate(items) if index > 0 and item.cut_before]
    ends = [start - 1 for start in starts[1:]] + [len(items) - 1]
    warmup = config.boundary_warmup_frames
    for start, end in zip(starts, ends):
        for index in range(start, min(end + 1, start + warmup)):
            items[index].motion_valid = False
            items[index].add_reason('motion_warmup_after_track_boundary')
        for index in range(max(start, end - warmup + 1), end + 1):
            items[index].motion_valid = False
            items[index].add_reason('motion_cooldown_before_track_boundary')

def assess_video_pose_quality(observations: Sequence[PoseQualityObservation], *, image_size: Sequence[float], config: QualityConfig | None=None) -> VideoQualityResult:
    """Assess all tracked poses from one video without ground truth.

    Returns per-frame branch masks and conservative segment boundaries.  Large
    coherent person motion is intentionally retained; only internal pose
    inconsistency affects the temporal corruption gate.
    """
    cfg = config or QualityConfig()
    cfg.validate()
    if len(image_size) != 2:
        raise ValueError('image_size must be (width, height)')
    resolved_size = (float(image_size[0]), float(image_size[1]))
    if not all((math.isfinite(value) and value > 0.0 for value in resolved_size)):
        raise ValueError('image_size values must be finite and positive')
    if not observations:
        return VideoQualityResult(tracks={})
    videos = {str(item.video) for item in observations}
    if len(videos) != 1:
        raise ValueError('assess_video_pose_quality accepts observations from one video')
    grouped: dict[tuple[str, str], list[_MutableQuality]] = {}
    by_frame: dict[int, list[_MutableQuality]] = {}
    seen: set[tuple[str, str, int]] = set()
    prepared_observations: Sequence[PoseQualityObservation] = observations
    for observation in prepared_observations:
        if observation.frame_index < 0:
            raise ValueError('frame_index must be non-negative')
        key = (str(observation.video), str(observation.track_id), int(observation.frame_index))
        if key in seen:
            raise ValueError(f'duplicate pose observation {key}')
        seen.add(key)
        item = _base_quality(observation, resolved_size, cfg)
        grouped.setdefault(key[:2], []).append(item)
        by_frame.setdefault(key[2], []).append(item)
    _apply_overlap_gates(by_frame, cfg)
    result: dict[tuple[str, str], tuple[FrameQuality, ...]] = {}
    for key, items in sorted(grouped.items()):
        items.sort(key=lambda item: item.observation.frame_index)
        _apply_temporal_gates(items, cfg)
        _apply_boundary_warmup(items, cfg)
        frames: list[FrameQuality] = []
        for item in items:
            severity = 'cut' if item.cut_before else 'branch_off' if not item.pose_valid or not item.motion_valid else 'ok'
            frames.append(FrameQuality(video=item.observation.video, track_id=item.observation.track_id, frame_index=item.observation.frame_index, pose_valid=item.pose_valid, motion_valid=item.motion_valid, cut_before=item.cut_before, severity=severity, reasons=tuple(item.reasons), unknown_reasons=tuple(item.unknown_reasons), metrics=dict(item.metrics), processed_keypoints=tuple((float(value) for value in item.keypoints.reshape(-1))), processed_bbox=item.bbox))
        result[key] = tuple(frames)
    return VideoQualityResult(tracks=result)

def observations_from_matching_records(records: Sequence[Mapping[str, Any]]) -> tuple[PoseQualityObservation, ...]:
    """Adapt matching JSON objects while ignoring every GT/label field.

    The only carried identity signal is the already-materialized
    ``suspected_split`` flag.  It is optional and absent during test inference;
    this function never examines why an official matcher created it.
    """
    observations: list[PoseQualityObservation] = []
    for record in records:
        keypoints = record.get('keypoints')
        if not isinstance(keypoints, (list, tuple, np.ndarray)):
            raise ValueError('matching record keypoints are required')
        bbox_value = record.get('predicted_bbox')
        bbox = None if bbox_value is None else tuple((float(value) for value in bbox_value))
        suspected = bool(record.get('suspected_split', False))
        observations.append(PoseQualityObservation(video=str(record.get('video', '')), track_id=str(record.get('predicted_track_id', '')), frame_index=int(record.get('frame_index')), keypoints=_as_keypoints(keypoints), predicted_bbox=bbox, existing_identity_break=suspected, existing_break_reason=str(record.get('split_reason')) if suspected and record.get('split_reason') is not None else None))
    return tuple(observations)

def assess_pose_records(records: Sequence[Mapping[str, Any]], *, image_size: Sequence[float], config: QualityConfig | None=None) -> VideoQualityResult:
    """Convenience adapter for current matching JSONL records."""
    return assess_video_pose_quality(observations_from_matching_records(records), image_size=image_size, config=config)
