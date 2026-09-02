"""Full-set frame metrics and an AED six-column region-score adapter.

The third-party AED evaluator and its data are not distributed here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .postprocess import _group_rows, _load_pose_people, _validate_frame_counts

def _arrays(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape:
        raise ValueError('labels and scores must be equal-length vectors')
    if not np.all((y == 0) | (y == 1)):
        raise ValueError('labels must be binary')
    if not np.isfinite(s).all():
        raise ValueError('scores must be finite')
    return (y.astype(np.int8), s)

def binary_auroc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Tie-aware ROC AUC using average ranks."""
    y, s = _arrays(labels, scores)
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return math.nan
    order = np.argsort(s, kind='mergesort')
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        stop = start + 1
        while stop < len(s) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = ranks[y == 1].sum()
    return float((positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))

def average_precision(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Non-interpolated average precision with tied scores grouped together."""
    y, s = _arrays(labels, scores)
    positive = int(y.sum())
    if positive == 0:
        return math.nan
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    s = s[order]
    cumulative_positive = np.cumsum(y)
    stop_indices = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    true_positive = cumulative_positive[stop_indices]
    predicted = stop_indices + 1
    previous = np.r_[0, true_positive[:-1]]
    recall_increment = (true_positive - previous) / positive
    precision = true_positive / predicted
    return float(np.sum(recall_increment * precision))

def keypoints_to_box(keypoints: Any, confidence_threshold: float=0.05, min_keypoints: int=3, padding_fraction: float=0.1) -> tuple[float, float, float, float] | None:
    """Enclose confident keypoints and pad each side by 10 percent."""
    points = np.asarray(keypoints, dtype=np.float64)
    if points.size % 3:
        raise ValueError(f'Keypoints length {points.size} is not divisible by three')
    points = points.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= float(confidence_threshold))
    if int(valid.sum()) < int(min_keypoints):
        return None
    xy = points[valid, :2]
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    if width <= 0 or height <= 0:
        return None
    xpad = float(padding_fraction) * width
    ypad = float(padding_fraction) * height
    return (max(0.0, float(xmin) - xpad), max(0.0, float(ymin) - ypad), float(xmax) + xpad, float(ymax) + ypad)

def _strict_frame_winner(rows: Iterable[dict[str, Any]]) -> str | None:
    by_track: dict[str, float] = {}
    for row in rows:
        score = float(row['gapfill_score'])
        track = str(row['track'])
        by_track[track] = max(by_track.get(track, -math.inf), score)
    positive = sorted(((score, track) for track, score in by_track.items() if score > 0.0), reverse=True)
    if not positive:
        return None
    if len(positive) > 1 and math.isclose(positive[0][0], positive[1][0], rel_tol=0.0, abs_tol=1e-12):
        return None
    return positive[0][1]

def infer_unique_context_track(frame: int, rows_by_frame: dict[int, list[dict[str, Any]]], target_tracks: set[str], context_frames: int=4) -> str | None:
    """Return one strict, stable winner seen on both sides of ``frame``."""
    sides: list[list[str]] = []
    for candidates in (range(max(0, frame - context_frames), frame), range(frame + 1, frame + context_frames + 1)):
        winners = [winner for candidate in candidates if (winner := _strict_frame_winner(rows_by_frame.get(candidate, ()))) is not None]
        if not winners or len(set(winners)) != 1:
            return None
        sides.append(winners)
    winner = sides[0][0]
    if sides[1][0] != winner or winner not in target_tracks:
        return None
    return winner

def redistribute_scores(rows_by_frame: dict[int, list[dict[str, Any]]], final_scores: np.ndarray, context_frames: int=4) -> dict[str, int]:
    """Mutate valid rows with ``aed_score`` and return assignment statistics."""
    statistics = {'scaled_frame_count': 0, 'zeroed_frame_count': 0, 'context_assigned_frame_count': 0, 'ambiguous_zero_base_frame_count': 0, 'positive_final_frame_without_rows_count': 0}
    for frame, final in enumerate(np.asarray(final_scores, dtype=np.float64)):
        if not math.isfinite(float(final)) or final < 0:
            raise ValueError(f'Invalid final score at frame {frame}: {final}')
        rows = rows_by_frame.get(frame, [])
        if not rows:
            if final > 0:
                statistics['positive_final_frame_without_rows_count'] += 1
            continue
        if final == 0:
            for row in rows:
                row['aed_score'] = 0.0
            statistics['zeroed_frame_count'] += 1
            continue
        maximum = max((float(row['gapfill_score']) for row in rows))
        if maximum > 0:
            scale = float(final) / maximum
            for row in rows:
                row['aed_score'] = float(row['gapfill_score']) * scale
            winners = [row for row in rows if row['gapfill_score'] == maximum]
            for row in winners:
                row['aed_score'] = float(final)
            statistics['scaled_frame_count'] += 1
            continue
        winner = infer_unique_context_track(frame, rows_by_frame, {str(row['track']) for row in rows}, context_frames=context_frames)
        for row in rows:
            row['aed_score'] = float(final) if row['track'] == winner else 0.0
        if winner is None:
            statistics['ambiguous_zero_base_frame_count'] += 1
        else:
            statistics['context_assigned_frame_count'] += 1
    return statistics


def evaluate_scores(
    scores: Mapping[str, Sequence[float]], labels: Mapping[str, Sequence[int]],
) -> dict[str, float | int]:
    """Compute micro AUROC/AP over every frame (0 normal, 1 abnormal).

    Video sets and vector lengths must agree exactly. Nothing is filtered,
    truncated, rescaled per video, or selected by its labels.
    """
    if not scores or set(scores) != set(labels):
        raise ValueError("score and label video sets must be identical and non-empty")
    all_scores, all_labels = [], []
    for video in sorted(scores):
        y, s = _arrays(labels[video], scores[video])
        if not len(y):
            raise ValueError(f"empty score/label vector: {video}")
        all_scores.append(s)
        all_labels.append(y)
    y, s = np.concatenate(all_labels), np.concatenate(all_scores)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("the complete evaluation split must contain both classes")
    return {
        "auroc": binary_auroc(y, s),
        "average_precision": average_precision(y, s),
        "frame_count": len(y),
        "normal_count": len(y) - positives,
        "abnormal_count": positives,
        "video_count": len(scores),
    }


def export_regions(
    person_rows: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]],
    frame_scores: Mapping[str, Sequence[float]],
    pose_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write AED rows ``frame,xmin,ymin,xmax,ymax,score`` for the full split.

    Frames are zero based. Boxes enclose confident keypoints with 10% padding.
    Final frame scores are redistributed over observed boxes, preserving the
    relative person scores. A positive score without an observed box does not
    create a region; an ambiguous all-zero base stays unassigned and is counted.
    This adapter exports predictions only, not the external AED metric code.
    """
    finals = {}
    for video, values in frame_scores.items():
        array = np.asarray(values, dtype=np.float64)
        if (
            array.ndim != 1
            or not len(array)
            or not np.isfinite(array).all()
            or np.any((array < 0) | (array > 1))
        ):
            raise ValueError(f"frame scores must be non-empty finite [0, 1] vectors: {video}")
        finals[video] = array
    counts = _validate_frame_counts({video: len(values) for video, values in finals.items()})
    grouped = _group_rows(person_rows, counts)
    pose_root, target = Path(pose_dir), Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite region output: {target}")
    for video in counts:
        path = pose_root / f"{video}_alphapose_tracked_person.json"
        if not path.is_file():
            raise FileNotFoundError(path)
    target.mkdir(parents=True)
    totals = {
        "video_count": len(counts), "input_person_row_count": 0,
        "prediction_row_count": 0, "pose_record_missing_count": 0,
        "keypoint_box_rejected_count": 0,
        "scaled_frame_count": 0, "zeroed_frame_count": 0,
        "context_assigned_frame_count": 0, "ambiguous_zero_base_frame_count": 0,
        "positive_final_frame_without_rows_count": 0,
    }
    videos = {}
    for video in sorted(counts):
        poses = _load_pose_people(pose_root / f"{video}_alphapose_tracked_person.json")
        by_frame: dict[int, list[dict[str, Any]]] = {}
        valid_rows = []
        local = {
            "input_person_row_count": len(grouped[video]),
            "prediction_row_count": 0, "pose_record_missing_count": 0,
            "keypoint_box_rejected_count": 0,
        }
        for source in grouped[video]:
            frame, track = source["frame_index"], source["track_id"]
            record = poses.get(track, {}).get(frame)
            if record is None:
                local["pose_record_missing_count"] += 1
                continue
            box = keypoints_to_box(record["keypoints"])
            if box is None:
                local["keypoint_box_rejected_count"] += 1
                continue
            score = float(source.get("processed_score", source["score"]))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid processed person score: {video}, {frame}, {track}")
            row = {"frame": frame, "track": track, "gapfill_score": score, "box": box}
            valid_rows.append(row)
            by_frame.setdefault(frame, []).append(row)
        local.update(redistribute_scores(by_frame, finals[video], context_frames=4))
        with (target / f"{video}.txt").open("w", encoding="utf-8", newline="\n") as stream:
            for row in sorted(valid_rows, key=lambda item: (item["frame"], int(item["track"]))):
                xmin, ymin, xmax, ymax = row["box"]
                stream.write(
                    f"{row['frame']},{xmin:.6f},{ymin:.6f},{xmax:.6f},"
                    f"{ymax:.6f},{row.get('aed_score', 0.0):.10g}\n"
                )
        local["prediction_row_count"] = len(valid_rows)
        videos[video] = local
        for name, value in local.items():
            totals[name] += value
    return {"totals": totals, "videos": videos}
