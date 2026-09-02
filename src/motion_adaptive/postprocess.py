"""Frozen identity-aware gap filling and frame despiking for UBnormal.

Only original observations anchor gap repairs. CHAD uses raw frame maxima.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

@dataclass(frozen=True)
class GapFillConfig:
    version: str = 'identity-aware-gap-fill-v2'
    calibration_boundary: str = 'train_validation_only'
    anchor_score: float = 0.85
    minimum_anchor_run: int = 2
    bad_pose_low_score_max: float = 0.5
    reliable_normal_score_max: float = 0.25
    reliable_normal_stop_run: int = 6
    max_bidirectional_gap: int = 50
    max_unilateral_gap: int = 50
    unilateral_decay: float = 0.99
    max_endpoint_center_distance: float = 0.75
    max_endpoint_log_scale_change: float = 0.7
    max_endpoint_pose_distance: float = 0.3
    minimum_common_pose_joints: int = 6
    cross_track_missing_pose_penalty: float = 0.35
    cross_track_second_best_margin: float = 0.15
    max_local_center_motion: float = 0.12
    max_local_log_scale_change: float = 0.18
    obvious_center_motion: float = 0.25
    obvious_log_scale_change: float = 0.3
    require_bbox: bool = True
    stop_at_image_edge: bool = True

    def validate(self) -> None:
        if self.calibration_boundary != 'train_validation_only':
            raise ValueError('score postprocess thresholds must be train/validation frozen')
        for name in ('anchor_score', 'bad_pose_low_score_max', 'reliable_normal_score_max'):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.reliable_normal_score_max > self.bad_pose_low_score_max:
            raise ValueError('reliable normal threshold cannot exceed bad-pose threshold')
        if self.minimum_anchor_run < 1:
            raise ValueError('minimum_anchor_run must be positive')
        if self.reliable_normal_stop_run < 1:
            raise ValueError('reliable_normal_stop_run must be positive')
        if not 1 <= self.max_bidirectional_gap <= 50:
            raise ValueError('max_bidirectional_gap must be in [1, 50]')
        if not 0 <= self.max_unilateral_gap <= 50:
            raise ValueError('max_unilateral_gap must be in [0, 50]')
        if not 0.0 < self.unilateral_decay <= 1.0:
            raise ValueError('unilateral_decay must be in (0, 1]')
        if self.minimum_common_pose_joints < 1:
            raise ValueError('minimum_common_pose_joints must be positive')
        for name in ('max_endpoint_center_distance', 'max_endpoint_log_scale_change', 'max_endpoint_pose_distance', 'cross_track_missing_pose_penalty', 'cross_track_second_best_margin', 'max_local_center_motion', 'max_local_log_scale_change', 'obvious_center_motion', 'obvious_log_scale_change'):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f'{name} must be non-negative')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class _GapFillResult:
    raw_frame_scores: tuple[float, ...]
    processed_frame_scores: tuple[float, ...]
    person_rows: tuple[dict[str, Any], ...]
    fill_events: tuple[dict[str, Any], ...]
    summary: Mapping[str, int | float | str]

def _finite_score(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        raise ValueError('person score must be finite or null')
    return score

def _bbox(row: Mapping[str, Any] | None) -> tuple[float, float, float, float] | None:
    if row is None:
        return None
    value = row.get('processed_bbox', row.get('pose_bbox', row.get('predicted_bbox')))
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = tuple((float(item) for item in value))
    if not all((math.isfinite(item) for item in box)):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box

def _box_change(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> tuple[float, float] | None:
    left_box, right_box = (_bbox(left), _bbox(right))
    if left_box is None or right_box is None:
        return None
    left_center = np.asarray([(left_box[0] + left_box[2]) * 0.5, (left_box[1] + left_box[3]) * 0.5])
    right_center = np.asarray([(right_box[0] + right_box[2]) * 0.5, (right_box[1] + right_box[3]) * 0.5])
    left_scale = max(1e-06, math.hypot(left_box[2] - left_box[0], left_box[3] - left_box[1]))
    right_scale = max(1e-06, math.hypot(right_box[2] - right_box[0], right_box[3] - right_box[1]))
    return (float(np.linalg.norm(right_center - left_center)) / max(1.0, min(left_scale, right_scale)), abs(math.log(right_scale / left_scale)))

def _pose(row: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    if row is None:
        return None
    value = row.get('processed_keypoints', row.get('pose_keypoints', row.get('_postprocess_keypoints', row.get('keypoints'))))
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size != 51:
        return None
    array = array.reshape(17, 3)
    box = _bbox(row)
    if box is None:
        return None
    valid = np.isfinite(array).all(axis=1) & (array[:, 2] >= 0.05)
    center = np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    scale = max(1.0, math.hypot(box[2] - box[0], box[3] - box[1]))
    return ((array[:, :2] - center) / scale, valid)

def _pose_distance(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None, minimum_common_joints: int) -> float | None:
    left_pose, right_pose = (_pose(left), _pose(right))
    if left_pose is None or right_pose is None:
        return None
    common = left_pose[1] & right_pose[1]
    if int(common.sum()) < minimum_common_joints:
        return None
    return float(np.median(np.linalg.norm(left_pose[0][common] - right_pose[0][common], axis=1)))

def _near_edge(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    if bool(row.get('near_image_edge', False)):
        return True
    metrics = row.get('quality_metrics')
    if isinstance(metrics, Mapping) and bool(metrics.get('near_image_edge', False)):
        return True
    reasons = {str(reason) for reason in row.get('quality_reasons', [])}
    return bool(reasons.intersection({'near_image_edge', 'edge_entry_exit_transition'}))
_BAD_POSE_REASONS = frozenset({'too_few_trusted_joints', 'low_joint_confidence', 'severely_incomplete_pose', 'sudden_bone_length_change', 'incoherent_joint_jump', 'left_right_label_swap_suspected', 'unstable_pose_envelope', 'severe_temporal_corruption_identity_break', 'severe_pose_overlap'})

def _bad_pose(row: Mapping[str, Any]) -> bool:
    if row.get('prediction_valid') is False or row.get('pose_valid') is False:
        return True
    reasons = {str(reason) for reason in row.get('quality_reasons', [])}
    return bool(reasons.intersection(_BAD_POSE_REASONS))

def _target(row: Mapping[str, Any] | None, config: GapFillConfig) -> bool:
    if row is None:
        return True
    if config.stop_at_image_edge and _near_edge(row):
        return False
    score = _finite_score(row.get('score'))
    if score is None:
        return True
    return score <= config.bad_pose_low_score_max

def _reliable_normal(row: Mapping[str, Any], config: GapFillConfig) -> bool:
    score = _finite_score(row.get('score'))
    return score is not None and score <= config.reliable_normal_score_max and (not _bad_pose(row))

def _original_anchor(row: Mapping[str, Any] | None, config: GapFillConfig) -> bool:
    if row is None or bool(row.get('score_filled', False)):
        return False
    score = _finite_score(row.get('score'))
    if score is None or score < config.anchor_score:
        return False
    if config.require_bbox and _bbox(row) is None:
        return False
    return not (config.stop_at_image_edge and _near_edge(row))

def _anchor_run(timeline: Mapping[int, Mapping[str, Any]], frame: int, direction: int, config: GapFillConfig) -> bool:
    anchors: list[Mapping[str, Any]] = []
    for offset in range(config.minimum_anchor_run):
        row = timeline.get(frame + direction * offset)
        if not _original_anchor(row, config):
            return False
        anchors.append(row)
    for first, second in zip(anchors, anchors[1:]):
        change = _box_change(first, second)
        if change is None or change[0] > config.max_local_center_motion or change[1] > config.max_local_log_scale_change:
            return False
    return True

def _continuous_endpoints(left: Mapping[str, Any], right: Mapping[str, Any], config: GapFillConfig, *, require_pose: bool) -> bool:
    change = _box_change(left, right)
    if change is None:
        return False
    if change[0] > config.max_endpoint_center_distance or change[1] > config.max_endpoint_log_scale_change:
        return False
    distance = _pose_distance(left, right, config.minimum_common_pose_joints)
    if require_pose and distance is None:
        return False
    return distance is None or distance <= config.max_endpoint_pose_distance

def _cross_track_cost(left: Mapping[str, Any], right: Mapping[str, Any], config: GapFillConfig) -> float | None:
    """Return a continuity cost; missing/bad pose lowers evidence, not eligibility."""
    change = _box_change(left, right)
    if change is None:
        return None
    if change[0] > config.max_endpoint_center_distance or change[1] > config.max_endpoint_log_scale_change:
        return None
    cost = change[0] / max(config.max_endpoint_center_distance, 1e-06) + change[1] / max(config.max_endpoint_log_scale_change, 1e-06)
    distance = None
    if not _bad_pose(left) and (not _bad_pose(right)):
        distance = _pose_distance(left, right, config.minimum_common_pose_joints)
    if distance is None:
        cost += config.cross_track_missing_pose_penalty
    else:
        if distance > config.max_endpoint_pose_distance:
            return None
        cost += distance / max(config.max_endpoint_pose_distance, 1e-06)
    return float(cost)

def _target_corridor(timeline: Mapping[int, Mapping[str, Any]], start: int, stop: int, config: GapFillConfig) -> bool:
    return all((_target(timeline.get(frame), config) for frame in range(start, stop + 1)))

def _contains_reliable_normal_run(timelines: Sequence[Mapping[int, Mapping[str, Any]]], start: int, stop: int, config: GapFillConfig) -> bool:
    count = 0
    for frame in range(start, stop + 1):
        rows = [timeline.get(frame) for timeline in timelines]
        originals = [row for row in rows if row is not None]
        if originals and all((_reliable_normal(row, config) for row in originals)):
            count += 1
            if count >= config.reliable_normal_stop_run:
                return True
        else:
            count = 0
    return False

def _event(*, video: str, track_id: str, frame: int, target: Mapping[str, Any] | None, score: float, reason: str, anchors: Sequence[tuple[str, int]], left_track_id: str | None=None, right_track_id: str | None=None) -> dict[str, Any]:
    return {'video': video, 'track_id': track_id, 'frame_index': int(frame), 'raw_score': None if target is None else _finite_score(target.get('score')), 'processed_score': float(min(1.0, max(0.0, score))), 'score_filled': True, 'fill_reason': reason, 'anchor_frames': [int(item[1]) for item in anchors], 'anchor_track_ids': [str(item[0]) for item in anchors], 'left_track_id': left_track_id, 'right_track_id': right_track_id, 'synthetic_observation': target is None, 'synthetic_values_are_anchors': False}

def _gap_fill(frame_scores: Sequence[float], person_rows: Sequence[Mapping[str, Any]], *, frame_count: int | None=None, config: GapFillConfig | None=None) -> _GapFillResult:
    cfg = config or GapFillConfig()
    cfg.validate()
    resolved_count = len(frame_scores) if frame_count is None else int(frame_count)
    raw = np.asarray(frame_scores, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != resolved_count or (not np.isfinite(raw).all()):
        raise ValueError('frame_scores must be a finite frame_count-length vector')
    if resolved_count < 1:
        raise ValueError('frame_count must be positive')
    videos = {str(row.get('video', '')) for row in person_rows}
    if len(videos) > 1:
        raise ValueError('gap filling accepts rows from one video')
    video = next(iter(videos), '')
    timelines: dict[str, dict[int, Mapping[str, Any]]] = {}
    augmented: list[dict[str, Any]] = []
    row_index: dict[tuple[str, int], int] = {}
    for source in person_rows:
        row = dict(source)
        track_id, frame = (str(row['track_id']), int(row['frame_index']))
        if not 0 <= frame < resolved_count:
            raise ValueError('person frame is outside the declared video')
        key = (track_id, frame)
        if key in row_index:
            raise ValueError(f'duplicate person-frame row: {key}')
        row['raw_score'] = _finite_score(row.get('score'))
        row['processed_score'] = row['raw_score']
        row['score_filled'] = False
        row['fill_reason'] = None
        row_index[key] = len(augmented)
        augmented.append(row)
        timelines.setdefault(track_id, {})[frame] = source
    events: list[dict[str, Any]] = []
    claimed: set[tuple[str, int]] = set()

    def add_fill(track_id: str, frame: int, score: float, reason: str, anchors: Sequence[tuple[str, int]], *, target_track_ids: Sequence[str]=(), left_track_id: str | None=None, right_track_id: str | None=None) -> None:
        identity = f'{left_track_id}->{right_track_id}' if left_track_id and right_track_id else track_id
        claim = (identity, frame)
        if claim in claimed:
            return
        candidates = [(candidate, timelines.get(candidate, {}).get(frame)) for candidate in target_track_ids or (track_id,)]
        existing = [(candidate, row) for candidate, row in candidates if row is not None and _target(row, cfg)]
        event_target = existing[0][1] if existing else None
        event_track = existing[0][0] if existing else identity
        item = _event(video=video, track_id=event_track, frame=frame, target=event_target, score=score, reason=reason, anchors=anchors, left_track_id=left_track_id, right_track_id=right_track_id)
        events.append(item)
        claimed.add(claim)
        for candidate, target in existing:
            index = row_index[candidate, frame]
            augmented[index]['processed_score'] = item['processed_score']
            augmented[index]['score_filled'] = True
            augmented[index]['fill_reason'] = reason
    for track_id, timeline in sorted(timelines.items()):
        targets = {frame for frame in range(resolved_count) if _target(timeline.get(frame), cfg)}
        for start in sorted(targets):
            if start - 1 in targets:
                continue
            stop = start
            while stop + 1 in targets:
                stop += 1
            length = stop - start + 1
            left_frame, right_frame = (start - 1, stop + 1)
            left, right = (timeline.get(left_frame), timeline.get(right_frame))
            if length <= cfg.max_bidirectional_gap and left is not None and (right is not None) and _anchor_run(timeline, left_frame, -1, cfg) and _anchor_run(timeline, right_frame, 1, cfg) and _continuous_endpoints(left, right, cfg, require_pose=False) and (not _contains_reliable_normal_run((timeline,), start, stop, cfg)):
                score = min(float(left['score']), float(right['score']))
                for frame in range(start, stop + 1):
                    add_fill(track_id, frame, score, 'bidirectional_same_track_v2', [(track_id, left_frame), (track_id, right_frame)])
    left_endpoints: list[tuple[str, int, Mapping[str, Any]]] = []
    right_endpoints: list[tuple[str, int, Mapping[str, Any]]] = []
    for track_id, timeline in sorted(timelines.items()):
        for frame, row in sorted(timeline.items()):
            if _anchor_run(timeline, frame, -1, cfg) and _target(timeline.get(frame + 1), cfg):
                left_endpoints.append((track_id, frame, row))
            if _anchor_run(timeline, frame, 1, cfg) and _target(timeline.get(frame - 1), cfg):
                right_endpoints.append((track_id, frame, row))
    candidate_pairs: list[tuple[tuple[str, int, Mapping[str, Any]], tuple[str, int, Mapping[str, Any]], float]] = []
    for left in left_endpoints:
        for right in right_endpoints:
            if left[0] == right[0]:
                continue
            gap = right[1] - left[1] - 1
            if not 1 <= gap <= cfg.max_bidirectional_gap:
                continue
            if not _target_corridor(timelines[left[0]], left[1] + 1, right[1] - 1, cfg):
                continue
            if not _target_corridor(timelines[right[0]], left[1] + 1, right[1] - 1, cfg):
                continue
            if _contains_reliable_normal_run((timelines[left[0]], timelines[right[0]]), left[1] + 1, right[1] - 1, cfg):
                continue
            cost = _cross_track_cost(left[2], right[2], cfg)
            if cost is not None:
                candidate_pairs.append((left, right, cost))
    by_left: dict[tuple[str, int], list[tuple[float, tuple[str, int]]]] = {}
    by_right: dict[tuple[str, int], list[tuple[float, tuple[str, int]]]] = {}
    for left, right, cost in candidate_pairs:
        by_left.setdefault((left[0], left[1]), []).append((cost, (right[0], right[1])))
        by_right.setdefault((right[0], right[1]), []).append((cost, (left[0], left[1])))

    def unique_best(candidates: Sequence[tuple[float, tuple[str, int]]], expected: tuple[str, int]) -> bool:
        ordered = sorted(candidates, key=lambda item: (item[0], item[1]))
        if not ordered or ordered[0][1] != expected:
            return False
        return len(ordered) == 1 or ordered[1][0] - ordered[0][0] >= cfg.cross_track_second_best_margin
    for left, right, _ in candidate_pairs:
        if not unique_best(by_left[left[0], left[1]], (right[0], right[1])):
            continue
        if not unique_best(by_right[right[0], right[1]], (left[0], left[1])):
            continue
        score = min(float(left[2]['score']), float(right[2]['score']))
        for frame in range(left[1] + 1, right[1]):
            add_fill(left[0], frame, score, 'bidirectional_unique_track_change_v2', [(left[0], left[1]), (right[0], right[1])], target_track_ids=(left[0], right[0]), left_track_id=left[0], right_track_id=right[0])
    for track_id, timeline in sorted(timelines.items()):
        for anchor_frame, anchor in sorted(timeline.items()):
            for direction in (-1, 1):
                if not _anchor_run(timeline, anchor_frame, -direction, cfg):
                    continue
                immediate = timeline.get(anchor_frame + direction)
                if not _target(immediate, cfg):
                    continue
                previous_original: Mapping[str, Any] = anchor
                pending_normal: list[tuple[int, Mapping[str, Any]]] = []
                planned: list[tuple[int, Mapping[str, Any] | None]] = []
                for distance in range(1, cfg.max_unilateral_gap + 1):
                    frame = anchor_frame + direction * distance
                    if not 0 <= frame < resolved_count:
                        break
                    row = timeline.get(frame)
                    if row is not None and cfg.stop_at_image_edge and _near_edge(row):
                        break
                    if row is not None:
                        change = _box_change(previous_original, row)
                        if change is not None and (change[0] > cfg.obvious_center_motion or change[1] > cfg.obvious_log_scale_change):
                            break
                    if row is not None and _reliable_normal(row, cfg):
                        pending_normal.append((frame, row))
                        if len(pending_normal) >= cfg.reliable_normal_stop_run:
                            pending_normal = []
                            break
                        previous_original = row
                        continue
                    if pending_normal:
                        planned.extend(pending_normal)
                        pending_normal = []
                    if _target(row, cfg):
                        planned.append((frame, row))
                        if row is not None:
                            previous_original = row
                        continue
                    if row is not None and _original_anchor(row, cfg):
                        previous_original = row
                        continue
                    break
                else:
                    planned.extend(pending_normal)
                    pending_normal = []
                if pending_normal:
                    planned.extend(pending_normal)
                for frame, _ in planned:
                    distance = abs(frame - anchor_frame)
                    add_fill(track_id, frame, float(anchor['score']) * cfg.unilateral_decay ** distance, 'unilateral_stable_anomaly_v2', [(track_id, anchor_frame)])
    processed = raw.copy()
    for item in events:
        frame = int(item['frame_index'])
        processed[frame] = max(processed[frame], float(item['processed_score']))
    events.sort(key=lambda item: (int(item['frame_index']), str(item['track_id']), str(item['fill_reason'])))
    changed = np.flatnonzero(processed != raw)
    return _GapFillResult(raw_frame_scores=tuple((float(value) for value in raw)), processed_frame_scores=tuple((float(value) for value in processed)), person_rows=tuple(augmented), fill_events=tuple(events), summary={'version': cfg.version, 'fill_event_count': len(events), 'synthetic_fill_count': sum((bool(item['synthetic_observation']) for item in events)), 'invalid_observation_fill_count': sum((not bool(item['synthetic_observation']) for item in events)), 'same_track_fill_count': sum((item['fill_reason'] == 'bidirectional_same_track_v2' for item in events)), 'cross_track_fill_count': sum((item['fill_reason'] == 'bidirectional_unique_track_change_v2' for item in events)), 'unilateral_fill_count': sum((item['fill_reason'] == 'unilateral_stable_anomaly_v2' for item in events)), 'changed_frame_count': int(len(changed)), 'maximum_score_increase': float(np.max(processed - raw)) if len(raw) else 0.0})

@dataclass(frozen=True)
class DespikeConfig:
    version: str = 'bidirectional-frame-score-despike-v1'
    calibration_boundary: str = 'train_validation_only'
    maximum_spike_width: int = 2
    stable_context_frames: int = 4
    minimum_spike_height: float = 0.3
    maximum_side_range: float = 0.1
    maximum_baseline_mean_difference: float = 0.1

    def validate(self) -> None:
        if self.calibration_boundary != 'train_validation_only':
            raise ValueError('despike thresholds must be train/validation frozen')
        if self.maximum_spike_width not in (1, 2):
            raise ValueError('maximum_spike_width must be 1 or 2')
        if self.stable_context_frames < 1:
            raise ValueError('stable_context_frames must be positive')
        for name in ('minimum_spike_height', 'maximum_side_range', 'maximum_baseline_mean_difference'):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class _DespikeResult:
    source_frame_scores: tuple[float, ...]
    despiked_frame_scores: tuple[float, ...]
    events: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, int | float | str]

def _despike(frame_scores: Sequence[float], *, config: DespikeConfig | None=None) -> _DespikeResult:
    """Replace isolated high and low runs using their stable two-sided baseline."""
    cfg = config or DespikeConfig()
    cfg.validate()
    source = np.asarray(frame_scores, dtype=np.float64)
    if source.ndim != 1 or source.size < 1 or (not np.isfinite(source).all()):
        raise ValueError('frame_scores must be a non-empty finite vector')
    if np.any((source < 0.0) | (source > 1.0)):
        raise ValueError('frame_scores must be in [0, 1]')
    context = cfg.stable_context_frames
    accepted: list[dict[str, Any]] = []
    occupied = np.zeros(source.size, dtype=bool)
    for width in range(cfg.maximum_spike_width, 0, -1):
        for start in range(context, source.size - context - width + 1):
            stop = start + width
            if occupied[start:stop].any():
                continue
            left = source[start - context:start]
            spike = source[start:stop]
            right = source[stop:stop + context]
            left_range = float(np.ptp(left))
            right_range = float(np.ptp(right))
            if left_range > cfg.maximum_side_range or right_range > cfg.maximum_side_range:
                continue
            left_mean = float(np.mean(left))
            right_mean = float(np.mean(right))
            if abs(left_mean - right_mean) > cfg.maximum_baseline_mean_difference:
                continue
            baseline = float(np.median(np.concatenate((left, right))))
            epsilon = 1e-12
            high = bool(np.all(spike - baseline > cfg.minimum_spike_height + epsilon))
            low = bool(np.all(baseline - spike > cfg.minimum_spike_height + epsilon))
            if not high and (not low):
                continue
            occupied[start:stop] = True
            accepted.append({'start_frame': start, 'end_frame': stop - 1, 'width': width, 'direction': 'high' if high else 'low', 'source_scores': [float(value) for value in spike], 'replacement_score': baseline, 'left_mean': left_mean, 'right_mean': right_mean, 'left_range': left_range, 'right_range': right_range})
    corrected = source.copy()
    for event in accepted:
        corrected[event['start_frame']:event['end_frame'] + 1] = event['replacement_score']
    changed = np.flatnonzero(np.abs(corrected - source) > 1e-12)
    high_events = sum((event['direction'] == 'high' for event in accepted))
    low_events = len(accepted) - high_events
    return _DespikeResult(source_frame_scores=tuple((float(value) for value in source)), despiked_frame_scores=tuple((float(value) for value in corrected)), events=tuple(accepted), summary={'version': cfg.version, 'event_count': len(accepted), 'high_spike_event_count': high_events, 'low_spike_event_count': low_events, 'changed_frame_count': int(changed.size), 'maximum_absolute_change': float(np.max(np.abs(corrected - source))) if changed.size else 0.0})

def _geometry_from_pose(keypoints: Any, threshold: float=0.05) -> list[float] | None:
    array = np.asarray(keypoints, dtype=np.float64)
    if array.size != 51:
        return None
    array = array.reshape(17, 3)
    valid = np.isfinite(array).all(axis=1) & (array[:, 2] >= threshold)
    if int(valid.sum()) < 2:
        return None
    low = np.quantile(array[valid, :2], 0.05, axis=0)
    high = np.quantile(array[valid, :2], 0.95, axis=0)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        return None
    return [float(low[0]), float(low[1]), float(high[0]), float(high[1])]

def _load_pose_people(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(raw, dict):
        raise TypeError(f'Pose JSON top level must be an object: {path}')
    people: dict[str, dict[int, dict[str, Any]]] = {}
    for raw_track, raw_frames in raw.items():
        track = str(int(raw_track))
        mappings = raw_frames if isinstance(raw_frames, list) else [raw_frames]
        frames: dict[int, dict[str, Any]] = {}
        for mapping in mappings:
            if not isinstance(mapping, dict):
                raise TypeError(f'Pose frames must be mappings: {path}, track={track}')
            for raw_frame, record in mapping.items():
                frame = int(raw_frame)
                if frame in frames:
                    raise ValueError(f'Duplicate pose frame: {path}, {track}, {frame}')
                if not isinstance(record, dict) or 'keypoints' not in record:
                    raise ValueError(f'Missing keypoints: {path}, {track}, {frame}')
                frames[frame] = record
        people[track] = frames
    return people


def _validate_frame_counts(frame_counts: Mapping[str, int]) -> dict[str, int]:
    if not frame_counts:
        raise ValueError("frame_counts must contain the complete evaluation split")
    counts = {}
    for video, count in frame_counts.items():
        if (
            not isinstance(video, str)
            or not video
            or video in {".", ".."}
            or any(character in video for character in "/\\:")
        ):
            raise ValueError("video IDs must be non-empty file stems")
        if isinstance(count, bool) or int(count) != count or int(count) < 1:
            raise ValueError(f"frame count must be a positive integer: {video}")
        counts[video] = int(count)
    return counts


def _group_rows(
    person_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]],
    frame_counts: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    counts = _validate_frame_counts(frame_counts)
    grouped: dict[str, list[dict[str, Any]]] = {video: [] for video in counts}
    if isinstance(person_rows, Mapping):
        if set(person_rows) - set(counts):
            raise ValueError("person rows contain videos outside the declared split")
        flat = []
        for video, rows in person_rows.items():
            for source in rows:
                row = dict(source)
                if "video" in row and row["video"] != video:
                    raise ValueError("person row video differs from its group")
                row["video"] = video
                flat.append(row)
    else:
        flat = person_rows
    seen = set()
    for source in flat:
        row = dict(source)
        video = str(row["video"])
        if video not in counts:
            raise ValueError(f"person row video is outside the declared split: {video}")
        frame = int(row["frame_index"])
        if frame != row["frame_index"] or not 0 <= frame < counts[video]:
            raise ValueError(f"person frame is outside the declared video: {video}, {frame}")
        track = str(int(row["track_id"]))
        identity = (video, frame, track)
        if identity in seen:
            raise ValueError(f"duplicate person-frame row: {identity}")
        seen.add(identity)
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"person score must be finite and in [0, 1]: {identity}")
        row.update(video=video, frame_index=frame, track_id=track, score=score)
        grouped[video].append(row)
    return grouped


def frame_max(
    person_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]],
    frame_counts: Mapping[str, int],
) -> dict[str, np.ndarray]:
    """Return raw frame maxima; absent/invalid people contribute zero.

    ``frame_counts`` must enumerate the full split, including pose-empty videos.
    This is the final aggregation for CHAD, which does not use gap repair.
    """
    counts = _validate_frame_counts(frame_counts)
    grouped = _group_rows(person_rows, counts)
    result = {}
    for video in sorted(counts):
        scores = np.zeros(counts[video], dtype=np.float32)
        for row in grouped[video]:
            if row.get("prediction_valid") is not False:
                frame = row["frame_index"]
                scores[frame] = max(scores[frame], row["score"])
        result[video] = scores
    return result


def postprocess_ubnormal(
    person_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]],
    pose_dir: str | Path,
    frame_counts: Mapping[str, int],
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Apply the frozen UBnormal gap-fill then despike pipeline.

    Rows use ``video``, ``track_id``, zero-based ``frame_index``, and ``score``.
    Retain the model's validity, bounding-box and quality fields: gap repair
    uses these observations, not labels. Optional ``config`` has ``gapfill``
    and ``despike`` objects; omitting it selects the released fixed parameters.

    The returned rows retain gap-filled person scores for region export.
    Synthetic gap events affect frame scores but do not invent person boxes.
    """
    settings = dict(config or {})
    if set(settings) - {"gapfill", "despike"}:
        raise ValueError("postprocess config only accepts gapfill and despike")
    gap_config = GapFillConfig(**settings.get("gapfill", {}))
    spike_config = DespikeConfig(**settings.get("despike", {}))
    gap_config.validate()
    spike_config.validate()
    counts = _validate_frame_counts(frame_counts)
    grouped = _group_rows(person_rows, counts)
    raw = frame_max(grouped, counts)
    pose_root = Path(pose_dir)
    if not pose_root.is_dir():
        raise FileNotFoundError(pose_root)
    scores, processed_rows = {}, []
    for video in sorted(counts):
        rows = grouped[video]
        if rows:
            people = _load_pose_people(pose_root / f"{video}_alphapose_tracked_person.json")
            for row in rows:
                observation = people.get(row["track_id"], {}).get(row["frame_index"])
                if observation is None:
                    continue
                keypoints = observation.get("keypoints")
                bbox = _geometry_from_pose(keypoints)
                if bbox is not None:
                    row.setdefault("processed_bbox", bbox)
                if isinstance(keypoints, list) and len(keypoints) == 51:
                    row.setdefault("_postprocess_keypoints", keypoints)
        gap = _gap_fill(raw[video], rows, frame_count=counts[video], config=gap_config)
        # Both stage boundaries are float32 in the released inference pipeline.
        repaired = np.asarray(gap.processed_frame_scores, dtype=np.float32)
        spike = _despike(repaired, config=spike_config)
        scores[video] = np.asarray(spike.despiked_frame_scores, dtype=np.float32)
        processed_rows.extend(
            {key: value for key, value in row.items() if not key.startswith("_postprocess_")}
            for row in gap.person_rows
        )
    return scores, processed_rows
