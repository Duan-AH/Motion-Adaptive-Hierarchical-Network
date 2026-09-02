"""Frame-bag supervision for the CHAD training protocol."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator
import numpy as np

@dataclass(frozen=True)
class MilTrainingView:
    labels: np.ndarray
    weights: np.ndarray
    fit_mask: np.ndarray
    report: dict[str, Any]

def _validated_arrays(bag_labels: np.ndarray, active: np.ndarray, video_indices: np.ndarray, frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(bag_labels)
    usable = np.asarray(active, dtype=bool)
    videos = np.asarray(video_indices)
    frames = np.asarray(frame_indices)
    if not (labels.ndim == usable.ndim == videos.ndim == frames.ndim == 1 and labels.shape == usable.shape == videos.shape == frames.shape):
        raise ValueError('MIL row arrays must be one-dimensional and shape-aligned')
    if labels.size == 0:
        raise ValueError('MIL row arrays must be non-empty')
    if not np.isin(labels, (0, 1)).all():
        raise ValueError('bag_labels must contain only zero and one')
    if not np.issubdtype(videos.dtype, np.integer) or not np.issubdtype(frames.dtype, np.integer):
        raise ValueError('video_indices and frame_indices must be integer arrays')
    if (videos < 0).any() or (frames < 0).any():
        raise ValueError('video_indices and frame_indices must be non-negative')
    return (labels.astype(np.uint8, copy=False), usable, videos.astype(np.int64, copy=False), frames.astype(np.int64, copy=False))

def _bag_rows(videos: np.ndarray, frames: np.ndarray) -> Iterator[np.ndarray]:
    row_number = np.arange(len(videos), dtype=np.int64)
    order = np.lexsort((row_number, frames, videos))
    sorted_videos = videos[order]
    sorted_frames = frames[order]
    starts = np.r_[0, np.flatnonzero((sorted_videos[1:] != sorted_videos[:-1]) | (sorted_frames[1:] != sorted_frames[:-1])) + 1]
    stops = np.r_[starts[1:], len(order)]
    for (start, stop) in zip(starts.tolist(), stops.tolist()):
        yield order[start:stop]

def _bag_label(labels: np.ndarray, rows: np.ndarray) -> int:
    values = np.unique(labels[rows])
    if values.shape != (1,):
        raise ValueError('frame label is not constant within a (video, frame) bag')
    return int(values[0])

def select_top1_positive_instances(probabilities: np.ndarray, bag_labels: np.ndarray, active: np.ndarray, video_indices: np.ndarray, frame_indices: np.ndarray, track_indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Select exactly one active row in every covered positive frame bag."""
    (labels, usable, videos, frames) = _validated_arrays(bag_labels, active, video_indices, frame_indices)
    scores = np.asarray(probabilities, dtype=np.float64)
    tracks = np.asarray(track_indices)
    if scores.shape != labels.shape or tracks.shape != labels.shape:
        raise ValueError('probabilities and track_indices must match MIL row shape')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('MIL probabilities must be finite and in [0, 1]')
    if not np.issubdtype(tracks.dtype, np.integer) or (tracks < 0).any():
        raise ValueError('track_indices must be non-negative integers')
    selected = np.zeros(labels.shape, dtype=bool)
    covered_positive = 0
    uncovered_positive = 0
    selected_scores: list[float] = []
    for rows in _bag_rows(videos, frames):
        if _bag_label(labels, rows) == 0:
            continue
        candidates = rows[usable[rows]]
        if candidates.size == 0:
            uncovered_positive += 1
            continue
        covered_positive += 1
        best = min(candidates.tolist(), key=lambda index: (-scores[index], int(tracks[index]), int(index)))
        selected[best] = True
        selected_scores.append(float(scores[best]))
    if int(selected.sum()) != covered_positive:
        raise RuntimeError('top-1 MIL selection count does not close')
    values = np.asarray(selected_scores, dtype=np.float64)
    report = {'policy': 'top1_probability_then_lowest_track_index_then_row_index', 'covered_positive_bag_count': covered_positive, 'uncovered_positive_bag_count_with_pose_rows': uncovered_positive, 'selected_count': int(selected.sum()), 'selected_probability': {'min': None if values.size == 0 else float(values.min()), 'median': None if values.size == 0 else float(np.median(values)), 'max': None if values.size == 0 else float(values.max())}}
    return (selected, report)

def build_mil_training_view(bag_labels: np.ndarray, active: np.ndarray, video_indices: np.ndarray, frame_indices: np.ndarray, *, selected_positive: np.ndarray | None) -> MilTrainingView:
    """Build row labels and weights for initialization or a hard-EM M-step.

    With ``selected_positive=None`` every active person shares its frame's one
    weight unit.  This is only the deterministic initializer.  With a mask,
    negative bags still share one unit while each positive bag contributes its
    one selected person and excludes every unselected person.
    """
    (labels, usable, videos, frames) = _validated_arrays(bag_labels, active, video_indices, frame_indices)
    selected = None
    if selected_positive is not None:
        selected = np.asarray(selected_positive, dtype=bool)
        if selected.shape != labels.shape:
            raise ValueError('selected_positive must match MIL row shape')
        if (selected & ~usable).any() or (selected & (labels == 0)).any():
            raise ValueError('selected_positive contains inactive or negative-bag rows')
    instance_labels = np.zeros(labels.shape, dtype=np.uint8)
    weights = np.zeros(labels.shape, dtype=np.float64)
    bag_counts = {'negative': 0, 'positive': 0, 'covered_negative': 0, 'covered_positive': 0, 'uncovered_negative_with_pose_rows': 0, 'uncovered_positive_with_pose_rows': 0}
    for rows in _bag_rows(videos, frames):
        label = _bag_label(labels, rows)
        name = 'positive' if label == 1 else 'negative'
        bag_counts[name] += 1
        candidates = rows[usable[rows]]
        if candidates.size == 0:
            bag_counts[f'uncovered_{name}_with_pose_rows'] += 1
            continue
        bag_counts[f'covered_{name}'] += 1
        if label == 0:
            weights[candidates] = 1.0 / float(candidates.size)
            continue
        if selected is None:
            instance_labels[candidates] = 1
            weights[candidates] = 1.0 / float(candidates.size)
            continue
        chosen = rows[selected[rows]]
        if chosen.shape != (1,):
            raise ValueError('every covered positive bag must have exactly one selected instance')
        instance_labels[chosen] = 1
        weights[chosen] = 1.0
    positive_weight = weights > 0
    if not positive_weight.any():
        raise ValueError('MIL view has no positive-weight rows')
    per_video_before: dict[str, float] = {}
    for video in np.unique(videos[positive_weight]):
        mask = positive_weight & (videos == video)
        total = float(weights[mask].sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f'video {int(video)} has invalid MIL base weight')
        per_video_before[str(int(video))] = total
        weights[mask] /= total
    class_before: dict[str, float] = {}
    for label in (0, 1):
        mask = positive_weight & (instance_labels == label)
        total = float(weights[mask].sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f'MIL class {label} has zero weight')
        class_before[str(label)] = total
        weights[mask] /= total
    mean_positive = float(weights[positive_weight].mean())
    if not np.isfinite(mean_positive) or mean_positive <= 0:
        raise ValueError('MIL positive weight mean is invalid')
    weights[positive_weight] /= mean_positive
    fit_mask = weights > 0
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise RuntimeError('MIL weights are not finite and non-negative')
    positive_values = weights[fit_mask]
    report = {'policy': 'equal_frame_then_equal_video_then_equal_class_then_positive_mean_one', 'mode': 'initializer' if selected is None else 'hard_em_top1', 'row_count': int(len(labels)), 'active_row_count': int(usable.sum()), 'fit_row_count': int(fit_mask.sum()), 'excluded_active_row_count': int((usable & ~fit_mask).sum()), 'bag_counts': bag_counts, 'fit_class_counts': {str(label): int((fit_mask & (instance_labels == label)).sum()) for label in (0, 1)}, 'per_video_frame_units_before_equalization': per_video_before, 'class_weight_before_equalization': class_before, 'final_class_weight': {str(label): float(weights[fit_mask & (instance_labels == label)].sum()) for label in (0, 1)}, 'positive_weight': {'min': float(positive_values.min()), 'median': float(np.median(positive_values)), 'max': float(positive_values.max()), 'mean': float(positive_values.mean())}}
    return MilTrainingView(labels=instance_labels, weights=weights, fit_mask=fit_mask, report=report)
