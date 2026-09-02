"""Prepared UBnormal supervision and label-blind pose loading."""
from __future__ import annotations
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .features import extract_person_features
from .motion import extract_multiscale_motion_evidence
from .orientation import extract_orientation_features
from .quality import PoseQualityObservation, assess_pose_records, assess_video_pose_quality

@dataclass
class PersonSequenceSample:
    """One reliable person-track segment.

    ``sample_id`` and ``frame_ids`` provide stable keys for a future matching
    stage.  Pair construction itself intentionally lives outside this module.
    Labels may contain ``-1`` for unknown frames; these are masked by default.
    """
    pose_features: Tensor
    motion_features: Tensor
    scene_id: int
    labels: Tensor | None = None
    label_mask: Tensor | None = None
    pose_valid: Tensor | None = None
    motion_valid: Tensor | None = None
    frame_ids: Tensor | None = None
    label_weights: Tensor | None = None
    sample_id: str = ''
    video: str = ''
    track_id: str = ''
    segment_id: str = ''

def _validated_sample(sample: PersonSequenceSample) -> PersonSequenceSample:
    if sample.pose_features.ndim != 2 or sample.motion_features.ndim != 2:
        raise ValueError('feature tensors must have shape [time, feature_dim]')
    length = sample.pose_features.shape[0]
    if length < 1 or sample.motion_features.shape[0] != length:
        raise ValueError('pose and motion sequences must have equal positive length')
    if sample.scene_id < 0:
        raise ValueError('scene_id must be non-negative')
    for name in ('labels', 'label_mask', 'pose_valid', 'motion_valid', 'frame_ids', 'label_weights'):
        value = getattr(sample, name)
        if value is not None and (value.ndim != 1 or value.shape[0] != length):
            raise ValueError(f'{name} must have shape [time]')
    if sample.labels is not None:
        known = sample.labels >= 0 if sample.label_mask is None else sample.label_mask.bool()
        known_values = sample.labels[known]
        if known_values.numel() and (not torch.all((known_values == 0) | (known_values == 1))):
            raise ValueError('known labels must be binary')
    if sample.label_weights is not None:
        if not torch.isfinite(sample.label_weights).all() or (sample.label_weights < 0).any():
            raise ValueError('label_weights must be finite and non-negative')
    return sample

def collate_person_sequences(samples: Sequence[PersonSequenceSample]) -> dict[str, object]:
    """Pad variable-length person segments into one model batch."""
    if not samples:
        raise ValueError('cannot collate an empty batch')
    checked = [_validated_sample(sample) for sample in samples]
    pose_dim = checked[0].pose_features.shape[1]
    motion_dim = checked[0].motion_features.shape[1]
    if any((sample.pose_features.shape[1] != pose_dim for sample in checked)):
        raise ValueError('all pose feature dimensions must match')
    if any((sample.motion_features.shape[1] != motion_dim for sample in checked)):
        raise ValueError('all motion feature dimensions must match')
    lengths = torch.tensor([sample.pose_features.shape[0] for sample in checked])
    maximum = int(lengths.max().item())
    time = torch.arange(maximum)
    sequence_mask = time.unsqueeze(0) < lengths.unsqueeze(1)

    def bool_sequence(value: Tensor | None, length: int, default: bool) -> Tensor:
        if value is None:
            return torch.full((length,), default, dtype=torch.bool)
        return value.to(dtype=torch.bool)
    labels = []
    label_masks = []
    pose_masks = []
    motion_masks = []
    frame_ids = []
    label_weights = []
    for sample in checked:
        length = sample.pose_features.shape[0]
        if sample.labels is None:
            labels.append(torch.zeros(length, dtype=torch.float32))
            label_masks.append(torch.zeros(length, dtype=torch.bool))
        else:
            labels.append(sample.labels.to(dtype=torch.float32).clamp_min(0))
            default_label_mask = sample.labels >= 0
            label_masks.append(default_label_mask if sample.label_mask is None else sample.label_mask.to(dtype=torch.bool) & default_label_mask)
        pose_masks.append(bool_sequence(sample.pose_valid, length, True))
        motion_masks.append(bool_sequence(sample.motion_valid, length, True))
        frame_ids.append(torch.arange(length, dtype=torch.long) if sample.frame_ids is None else sample.frame_ids.to(dtype=torch.long))
        label_weights.append(torch.ones(length, dtype=torch.float32) if sample.label_weights is None else sample.label_weights.to(dtype=torch.float32))
    return {'pose_features': pad_sequence([sample.pose_features for sample in checked], batch_first=True), 'motion_features': pad_sequence([sample.motion_features for sample in checked], batch_first=True), 'scene_ids': torch.tensor([sample.scene_id for sample in checked], dtype=torch.long), 'labels': pad_sequence(labels, batch_first=True), 'label_mask': pad_sequence(label_masks, batch_first=True), 'pose_valid': pad_sequence(pose_masks, batch_first=True), 'motion_valid': pad_sequence(motion_masks, batch_first=True), 'frame_ids': pad_sequence(frame_ids, batch_first=True, padding_value=-1), 'label_weights': pad_sequence(label_weights, batch_first=True), 'sequence_mask': sequence_mask, 'lengths': lengths, 'sample_ids': [sample.sample_id for sample in checked], 'videos': [sample.video for sample in checked], 'track_ids': [sample.track_id for sample in checked], 'segment_ids': [sample.segment_id for sample in checked]}

_SCENE_PATTERN = re.compile('(?:^|_)scene_(\\d+)(?:_|$)', re.IGNORECASE)

_LABEL_VALUES = {'normal': 0.0, 'abnormal': 1.0, 'unknown': -1.0}

FILTER_RULE_ID = 'reliable-anchor-propagated-no-conflict-v1'

RELIABLE_CONFIDENCE_TIERS = frozenset({'anchor', 'propagated'})

class TrainingDataError(ValueError):
    """A matching record cannot safely enter the training-data contract."""

@dataclass(frozen=True)
class TrainingRecordFilterReport:
    total_count: int = 0
    retained_count: int = 0
    removed_count: int = 0
    retained_normal_count: int = 0
    retained_abnormal_count: int = 0
    removed_normal_count: int = 0
    removed_abnormal_count: int = 0
    removed_unknown_count: int = 0
    removed_unreliable_tier_count: int = 0
    removed_suspected_split_count: int = 0

def filter_reliable_training_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], TrainingRecordFilterReport]:
    """Keep only known, reliable, non-conflicting training observations.

    Removed observations are not converted to masked timesteps.  Their absence
    therefore creates a real frame gap, which ``_continuous_runs`` uses to cut
    the person sequence.
    """
    retained: list[dict[str, Any]] = []
    counts = {'total_count': len(records), 'retained_count': 0, 'removed_count': 0, 'retained_normal_count': 0, 'retained_abnormal_count': 0, 'removed_normal_count': 0, 'removed_abnormal_count': 0, 'removed_unknown_count': 0, 'removed_unreliable_tier_count': 0, 'removed_suspected_split_count': 0}
    for raw in records:
        label = str(raw.get('label', '')).strip().lower()
        if label not in _LABEL_VALUES:
            line_number = int(raw.get('_jsonl_line', -1))
            raise TrainingDataError(f'label must be normal, abnormal, or unknown at JSONL line {line_number}')
        tier = str(raw.get('confidence_tier', '')).strip().lower()
        unreliable_tier = tier not in RELIABLE_CONFIDENCE_TIERS
        suspected_split = raw.get('suspected_split') is not False
        keep = label != 'unknown' and (not unreliable_tier) and (not suspected_split)
        if keep:
            retained.append(dict(raw))
            counts['retained_count'] += 1
            counts[f'retained_{label}_count'] += 1
        else:
            counts['removed_count'] += 1
            counts[f'removed_{label}_count'] += 1
            if unreliable_tier:
                counts['removed_unreliable_tier_count'] += 1
            if suspected_split:
                counts['removed_suspected_split_count'] += 1
    return (retained, TrainingRecordFilterReport(**counts))

@dataclass(frozen=True)
class TrainingSequenceBuild:
    samples: tuple[PersonSequenceSample, ...]
    record_count: int
    known_label_count: int
    unknown_label_count: int
    filter_report: TrainingRecordFilterReport

def parse_scene_number(video: str) -> int:
    """Return the one-based UBnormal scene number without reading scenario ID."""
    match = _SCENE_PATTERN.search(video)
    if match is None:
        raise TrainingDataError(f'cannot parse scene number from video {video!r}')
    scene_number = int(match.group(1))
    if scene_number < 1:
        raise TrainingDataError('scene number must be positive')
    return scene_number

def load_matching_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects while retaining their physical line number."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[dict[str, Any]] = []
    with source.open('r', encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f'invalid JSON at {source}:{line_number}: {exc.msg}') from exc
            if not isinstance(value, dict):
                raise TrainingDataError(f'matching record must be an object at {source}:{line_number}')
            value = dict(value)
            value['_jsonl_line'] = line_number
            records.append(value)
    if not records:
        raise TrainingDataError(f'matching JSONL contains no records: {source}')
    return records

def _finite_number(value: Any, field: str, line_number: int) -> float:
    if isinstance(value, bool):
        raise TrainingDataError(f'{field} must be numeric at JSONL line {line_number}')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingDataError(f'{field} must be numeric at JSONL line {line_number}') from exc
    if not math.isfinite(result):
        raise TrainingDataError(f'{field} must be finite at JSONL line {line_number}')
    return result

def _prepared_record(record: Mapping[str, Any]) -> dict[str, Any]:
    line_number = int(record.get('_jsonl_line', -1))
    video = str(record.get('video', '')).strip()
    segment_id = str(record.get('segment_id', '')).strip()
    track_id = str(record.get('predicted_track_id', '')).strip()
    if not video or not segment_id or (not track_id):
        raise TrainingDataError(f'video, segment_id, and predicted_track_id are required at JSONL line {line_number}')
    frame_value = record.get('frame_index')
    if isinstance(frame_value, bool):
        raise TrainingDataError(f'frame_index must be an integer at JSONL line {line_number}')
    try:
        frame_index = int(frame_value)
    except (TypeError, ValueError) as exc:
        raise TrainingDataError(f'frame_index must be an integer at JSONL line {line_number}') from exc
    if frame_index < 0 or frame_value != frame_index:
        raise TrainingDataError(f'frame_index must be a non-negative integer at JSONL line {line_number}')
    label_name = str(record.get('label', '')).strip().lower()
    if label_name not in _LABEL_VALUES:
        raise TrainingDataError(f'label must be normal, abnormal, or unknown at JSONL line {line_number}')
    confidence = _finite_number(record.get('confidence', 0.0), 'confidence', line_number)
    if not 0.0 <= confidence <= 1.0:
        raise TrainingDataError(f'confidence must be in [0, 1] at JSONL line {line_number}')
    raw_keypoints = record.get('keypoints')
    if not isinstance(raw_keypoints, list) or len(raw_keypoints) != 51:
        raise TrainingDataError(f'keypoints must contain 17 x,y,confidence triples at JSONL line {line_number}')
    keypoints = torch.tensor([_finite_number(value, 'keypoints', line_number) for value in raw_keypoints], dtype=torch.float32).reshape(17, 3)
    return {'video': video, 'scene_number': parse_scene_number(video), 'segment_id': segment_id, 'track_id': track_id, 'frame_index': frame_index, 'label': _LABEL_VALUES[label_name], 'label_known': label_name != 'unknown', 'label_weight': confidence if label_name != 'unknown' else 0.0, 'keypoints': keypoints, 'quality_motion_valid': bool(record.get('_quality_motion_valid', True)), 'line_number': line_number}

def _continuous_runs(records: Sequence[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    current: list[dict[str, Any]] = []
    previous_frame: int | None = None
    for record in sorted(records, key=lambda item: item['frame_index']):
        frame = int(record['frame_index'])
        if previous_frame is not None and frame <= previous_frame:
            if frame == previous_frame:
                raise TrainingDataError(f"duplicate frame {frame} in segment {record['segment_id']!r}")
            raise RuntimeError('records were not sorted')
        if current and frame != previous_frame + 1:
            yield current
            current = []
        current.append(record)
        previous_frame = frame
    if current:
        yield current

def build_person_sequences(records: Sequence[Mapping[str, Any]], *, image_size: Sequence[float], scene_id_base: int=1, confidence_threshold: float=0.05, min_visible_joints: int=5, light_smoothing: int=3, strong_smoothing: int=9) -> TrainingSequenceBuild:
    """Build one sample per declared segment and actual contiguous run.

    Only reliable, known, non-conflicting records enter the sequence.  Removed
    observations create frame gaps and therefore cut temporal runs.  Neither a
    declared segment nor a frame gap is ever crossed.
    """
    filtered, filter_report = filter_reliable_training_records(records)
    prepared = [_prepared_record(record) for record in filtered]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in prepared:
        key = (record['video'], record['track_id'], record['segment_id'])
        groups.setdefault(key, []).append(record)
    samples: list[PersonSequenceSample] = []
    for (video, track_id, segment_id), grouped in sorted(groups.items()):
        scene_numbers = {record['scene_number'] for record in grouped}
        if len(scene_numbers) != 1:
            raise TrainingDataError(f'mixed scenes in segment {segment_id!r}')
        for run_index, run in enumerate(_continuous_runs(grouped)):
            keypoints = torch.stack([record['keypoints'] for record in run]).unsqueeze(0)
            frame_ids = torch.tensor([record['frame_index'] for record in run], dtype=torch.long)
            quality_motion_valid = torch.tensor([record['quality_motion_valid'] for record in run], dtype=torch.bool).unsqueeze(0)
            features = extract_person_features(keypoints, image_size=image_size, confidence_threshold=confidence_threshold, min_visible_joints=min_visible_joints, light_smoothing=light_smoothing, strong_smoothing=strong_smoothing)
            motion_features = features.motion_shape[0]
            motion_valid = features.motion_valid[0]
            evidence = extract_multiscale_motion_evidence(keypoints, frame_ids=frame_ids.unsqueeze(0), sequence_mask=torch.ones((1, len(run)), dtype=torch.bool, device=keypoints.device), motion_valid=features.motion_valid & quality_motion_valid, confidence_threshold=confidence_threshold, min_visible_joints=min_visible_joints)
            orientation = extract_orientation_features(keypoints, frame_ids=frame_ids.unsqueeze(0), sequence_mask=torch.ones((1, len(run)), dtype=torch.bool, device=keypoints.device), confidence_threshold=confidence_threshold)
            motion_features = torch.cat((features.motion_shape[0], evidence.values[0], torch.stack((orientation.score[0], orientation.reliability[0]), dim=-1)), dim=-1)
            motion_valid = features.motion_valid[0] & evidence.component_valid[0].all(dim=-1)
            labels = torch.tensor([record['label'] for record in run], dtype=torch.float32)
            label_mask = torch.tensor([record['label_known'] for record in run], dtype=torch.bool)
            label_weights = torch.tensor([record['label_weight'] for record in run], dtype=torch.float32)
            samples.append(PersonSequenceSample(pose_features=features.pose[0], motion_features=motion_features, scene_id=next(iter(scene_numbers)) - scene_id_base, labels=labels, label_mask=label_mask, pose_valid=features.pose_valid[0], motion_valid=motion_valid, frame_ids=frame_ids, label_weights=label_weights, sample_id=f'{segment_id}:run{run_index:03d}', video=video, track_id=track_id, segment_id=segment_id))
    return TrainingSequenceBuild(samples=tuple(samples), record_count=len(prepared), known_label_count=sum((record['label_known'] for record in prepared)), unknown_label_count=sum((not record['label_known'] for record in prepared)), filter_report=filter_report)

def slice_sample(sample: PersonSequenceSample, start: int, stop: int) -> PersonSequenceSample:
    if not 0 <= start < stop <= sample.pose_features.shape[0]:
        raise ValueError('invalid sample slice')

    def take(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value[start:stop]
    return replace(sample, pose_features=sample.pose_features[start:stop], motion_features=sample.motion_features[start:stop], labels=take(sample.labels), label_mask=take(sample.label_mask), pose_valid=take(sample.pose_valid), motion_valid=take(sample.motion_valid), frame_ids=take(sample.frame_ids), label_weights=take(sample.label_weights), sample_id=f'{sample.sample_id}@{start}:{stop}')

class SequenceWindowDataset(Dataset[PersonSequenceSample]):
    """Random train crops or deterministic non-overlapping eval chunks."""

    def __init__(self, samples: Sequence[PersonSequenceSample], *, max_frames: int, mode: str, seed: int=0) -> None:
        if not samples:
            raise ValueError('samples must not be empty')
        if max_frames < 1:
            raise ValueError('max_frames must be positive')
        if mode not in {'train', 'eval'}:
            raise ValueError('mode must be train or eval')
        self.samples = tuple(samples)
        self.max_frames = int(max_frames)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0
        self._eval_windows: list[tuple[int, int, int]] = []
        if mode == 'eval':
            for sample_index, sample in enumerate(self.samples):
                length = sample.pose_features.shape[0]
                for start in range(0, length, self.max_frames):
                    self._eval_windows.append((sample_index, start, min(length, start + self.max_frames)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples) if self.mode == 'train' else len(self._eval_windows)

    def __getitem__(self, index: int) -> PersonSequenceSample:
        if self.mode == 'eval':
            sample_index, start, stop = self._eval_windows[index]
            return slice_sample(self.samples[sample_index], start, stop)
        sample = self.samples[index]
        length = sample.pose_features.shape[0]
        if length <= self.max_frames:
            return sample
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + index * 97)
        start = int(torch.randint(length - self.max_frames + 1, (), generator=generator))
        return slice_sample(sample, start, start + self.max_frames)

def class_counts(samples: Iterable[PersonSequenceSample]) -> tuple[int, int]:
    negatives = positives = 0
    for sample in samples:
        if sample.labels is None:
            continue
        mask = sample.labels >= 0
        if sample.label_mask is not None:
            mask &= sample.label_mask.bool()
        labels = sample.labels[mask]
        negatives += int((labels == 0).sum())
        positives += int((labels == 1).sum())
    return (negatives, positives)

def positive_class_weight(samples: Iterable[PersonSequenceSample]) -> float:
    negatives, positives = class_counts(samples)
    if negatives == 0 or positives == 0:
        raise ValueError('both normal and abnormal labels are required')
    return negatives / positives

POSE_SUFFIX = '_alphapose_tracked_person.json'

@dataclass(frozen=True)
class InferenceVideo:
    video: str
    frame_count: int
    samples: tuple[PersonSequenceSample, ...]
    quality_lookup: Mapping[tuple[str, str, int], Any]

def video_name_from_pose_path(path: str | Path) -> str:
    name = Path(path).name
    if not name.endswith(POSE_SUFFIX):
        raise ValueError(f'pose filename must end with {POSE_SUFFIX!r}: {name}')
    return name[:-len(POSE_SUFFIX)]

def _load_pose_observations(path: Path) -> tuple[str, list[PoseQualityObservation]]:
    video = video_name_from_pose_path(path)
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'pose root is not an object: {path}')
    observations: list[PoseQualityObservation] = []
    for raw_track, raw_frames in value.items():
        if not isinstance(raw_frames, dict):
            raise ValueError(f'track {raw_track!r} is not an object in {path}')
        for raw_frame, item in raw_frames.items():
            if not isinstance(item, dict) or 'keypoints' not in item:
                raise ValueError(f'invalid pose item {raw_track}/{raw_frame} in {path}')
            keypoints = np.asarray(item['keypoints'], dtype=np.float32)
            if keypoints.size != 51:
                raise ValueError(f'pose must contain 17 triples at {raw_track}/{raw_frame}')
            observations.append(PoseQualityObservation(video=video, track_id=str(raw_track), frame_index=int(raw_frame), keypoints=keypoints.reshape(17, 3)))
    return (video, observations)

def _runs(frames: Sequence[Any]) -> Iterable[list[Any]]:
    current: list[Any] = []
    previous: int | None = None
    for item in frames:
        frame = int(item.frame_index)
        if current and (item.cut_before or frame != previous + 1):
            yield current
            current = []
        current.append(item)
        previous = frame
    if current:
        yield current

def _load_pose_video(pose_path: str | Path, *, frame_count: int, image_size: Sequence[float]) -> InferenceVideo:
    """Load test pose and quality.  No masks, official IDs or labels are inputs."""
    path = Path(pose_path)
    video, observations = _load_pose_observations(path)
    if any((item.frame_index >= frame_count for item in observations)):
        raise ValueError(f'pose frame is outside declared video length for {video}')
    if not observations:
        return InferenceVideo(video, frame_count, (), {})
    quality = assess_video_pose_quality(observations, image_size=image_size)
    samples: list[PersonSequenceSample] = []
    for (_, track_id), frames in sorted(quality.tracks.items()):
        for run_index, run in enumerate(_runs(frames)):
            keypoints = torch.tensor(np.stack([np.asarray(item.processed_keypoints, dtype=np.float32).reshape(17, 3) for item in run]), dtype=torch.float32).unsqueeze(0)
            pose_valid = torch.tensor([item.pose_valid for item in run], dtype=torch.bool)
            motion_valid = torch.tensor([item.motion_valid for item in run], dtype=torch.bool)
            frame_ids = torch.tensor([item.frame_index for item in run], dtype=torch.long)
            features = extract_person_features(keypoints, image_size=image_size)
            motion_features = features.motion_shape[0]
            evidence = extract_multiscale_motion_evidence(keypoints, frame_ids=frame_ids.unsqueeze(0), sequence_mask=torch.ones((1, len(run)), dtype=torch.bool, device=keypoints.device), motion_valid=motion_valid.unsqueeze(0))
            orientation = extract_orientation_features(keypoints, frame_ids=frame_ids.unsqueeze(0), sequence_mask=torch.ones((1, len(run)), dtype=torch.bool, device=keypoints.device))
            motion_features = torch.cat((features.motion_shape[0], evidence.values[0], torch.stack((orientation.score[0], orientation.reliability[0]), dim=-1)), dim=-1)
            motion_valid = motion_valid & evidence.component_valid[0].all(dim=-1)
            samples.append(PersonSequenceSample(pose_features=features.pose[0], motion_features=motion_features, scene_id=parse_scene_number(video) - 1, pose_valid=pose_valid, motion_valid=motion_valid, frame_ids=frame_ids, sample_id=f'{video}:{track_id}:run{run_index:03d}', video=video, track_id=track_id, segment_id=f'{video}:{track_id}:run{run_index:03d}'))
    return InferenceVideo(video, frame_count, tuple(samples), quality.frame_lookup)

def _quality_records(records, image_size):
    result = assess_pose_records(records, image_size=image_size)
    lookup = dict(result.frame_lookup)
    revised, cut_index = ([], {})
    for raw in sorted(records, key=lambda r: (str(r.get('video', '')), str(r.get('predicted_track_id', '')), int(r.get('frame_index', -1)))):
        row = dict(raw)
        key = (str(row.get('video', '')), str(row.get('predicted_track_id', '')), int(row.get('frame_index', -1)))
        quality = lookup[key]
        base = str(row.get('segment_id', ''))
        group = (key[0], key[1], base)
        if quality.cut_before:
            cut_index[group] = cut_index.get(group, 0) + 1
        row['segment_id'] = f'{base}:q{cut_index.get(group, 0):03d}'
        row['_quality_motion_valid'] = bool(quality.motion_valid)
        revised.append(row)
    return (revised, lookup)

def _image_size(entry):
    size = entry.get('image_size', (entry.get('width'), entry.get('height')))
    if len(size) != 2 or any((isinstance(v, bool) or v is None or int(v) != v or (int(v) < 1) for v in size)):
        raise ValueError('metadata requires positive integer width and height')
    return tuple(map(int, size))

def load_ub_split(matching_dir: str | Path, metadata: Mapping[str, Any], *, expected_split: str | None = None) -> tuple[PersonSequenceSample, ...]:
    """Load prepared training/validation matching records; never follow embedded paths."""
    metadata = metadata.get('videos', metadata)
    paths = tuple(sorted((Path(matching_dir) / 'videos').glob('*.jsonl')))
    if not paths:
        raise FileNotFoundError('no prepared matching JSONL files in videos/')
    if {path.stem for path in paths} != set(metadata):
        raise ValueError('matching files and complete metadata video inventories differ')
    samples, videos = ([], set())
    for path in paths:
        records = load_matching_jsonl(path)
        if expected_split is not None and any(row.get('split') != expected_split for row in records):
            raise ValueError(f'matching records do not belong to {expected_split}: {path.name}')
        video_names = {str(r['video']) for r in records}
        if len(video_names) != 1:
            raise ValueError('one video per matching JSONL is required')
        video = video_names.pop()
        if video != path.stem:
            raise ValueError('matching filename and contained video differ')
        if video in videos:
            raise ValueError('duplicate matching video')
        if video not in metadata:
            raise ValueError(f'matching video is absent from metadata: {video}')
        videos.add(video)
        image_size = _image_size(metadata[video])
        filtered, _ = filter_reliable_training_records(records)
        if not filtered:
            continue
        quality_rows, lookup = _quality_records(filtered, image_size)
        for sample in build_person_sequences(quality_rows, image_size=image_size).samples:
            pose = torch.tensor([lookup[sample.video, sample.track_id, int(f)].pose_valid for f in sample.frame_ids], dtype=torch.bool)
            motion = torch.tensor([lookup[sample.video, sample.track_id, int(f)].motion_valid for f in sample.frame_ids], dtype=torch.bool)
            samples.append(replace(sample, pose_valid=sample.pose_valid & pose, motion_valid=sample.motion_valid & motion))
    if videos != set(metadata):
        raise ValueError('matching files and complete metadata video inventories differ')
    return tuple(samples)

def load_pose_video(pose_path: str | Path, metadata_entry: Mapping[str, Any]) -> InferenceVideo:
    count = metadata_entry['frame_count']
    if isinstance(count, bool) or int(count) != count or int(count) < 1:
        raise ValueError('metadata requires positive integer frame_count')
    return _load_pose_video(pose_path, frame_count=int(count), image_size=_image_size(metadata_entry))
