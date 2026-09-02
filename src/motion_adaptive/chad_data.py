"""CHAD Split 2 pose sequences and disk-backed frame-MIL views.

Only trusted official annotations may be opened: CHAD uses Python pickle.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import PersonSequenceSample, _quality_records, build_person_sequences
FEATURE_PROFILE = "pose85_motion31_lags3_9_15"

CACHE_SCHEMA = 'chad-motion-paper-p85-m31-cache-v1'

ARRAYS = ('pose', 'motion', 'pose_valid', 'motion_valid', 'frame', 'track')

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return value

def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def _pid_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.lstrip('-').isdigit() else (1, text)

def records_from_chad(info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reuse the established CHAD geometry semantics without opening labels.

    Dummy normal/anchor fields are transport-only for the existing builder,
    never training targets.  Real labels are attached after feature building.
    """
    with Path(info['annotation']).open('rb') as stream:
        document = pickle.load(stream)
    count = int(info['frame_count'])
    if not isinstance(document, dict) or list(document) != list(range(count)):
        raise ValueError(f"non-contiguous CHAD frame inventory: {info['name']}")
    width, height = map(int, info['image_size'])
    records: list[dict[str, Any]] = []
    for frame, people in document.items():
        for person_id in sorted(people, key=_pid_key):
            box, stored = people[person_id]
            box = np.asarray(box, dtype=np.float64)
            pose = np.asarray(stored, dtype=np.float64)
            if box.shape != (4,) or pose.shape != (17, 3):
                raise ValueError('CHAD box/pose shape changed')
            if not np.isfinite(box).all() or not np.isfinite(pose[:, :2]).all():
                raise ValueError('non-finite CHAD coordinates')
            if (box[2:] <= 0).any():
                raise ValueError('non-positive CHAD box size')
            confidence = pose[:, 2].copy()
            if np.isnan(confidence).all():
                confidence[:] = 1.0
            elif not np.isfinite(confidence).all():
                raise ValueError('mixed finite/non-finite CHAD confidence')
            points = np.column_stack((pose[:, 1], pose[:, 0], np.clip(confidence, 0, 1)))
            x, y, bw, bh = box
            clipped = [float(np.clip(x, 0, width)), float(np.clip(y, 0, height)), float(np.clip(x + bw, 0, width)), float(np.clip(y + bh, 0, height))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError('CHAD box has no image intersection')
            track = str(person_id)
            records.append({'video': info['canonical_video'], 'segment_id': f'chad-track-{track}', 'predicted_track_id': track, 'frame_index': int(frame), 'label': 'normal', 'confidence': 1.0, 'confidence_tier': 'anchor', 'suspected_split': False, 'predicted_bbox': clipped, 'keypoints': points.astype(np.float32).reshape(-1).tolist()})
    if len(records) != int(info['row_count']):
        raise ValueError('CHAD raw person-row count changed')
    return records

def build_video_sequences(info: Mapping[str, Any]) -> tuple[PersonSequenceSample, ...]:
    records = records_from_chad(info)
    if not records:
        return ()
    quality_rows, lookup = _quality_records(records, info['image_size'])
    built = build_person_sequences(quality_rows, image_size=info['image_size'])
    samples = []
    for sample in built.samples:
        frames = sample.frame_ids.tolist()
        qpose = torch.tensor([lookup[sample.video, sample.track_id, int(f)].pose_valid for f in frames])
        qmotion = torch.tensor([lookup[sample.video, sample.track_id, int(f)].motion_valid for f in frames])
        samples.append(replace(sample, pose_valid=sample.pose_valid & qpose, motion_valid=sample.motion_valid & qmotion, labels=None, label_mask=None, label_weights=None))
    if sum((len(s.frame_ids) for s in samples)) != int(info['row_count']):
        raise RuntimeError('label-blind sequence building dropped raw person rows')
    return tuple(samples)

def _video_cache(cache: Path, info: Mapping[str, Any], split: str, builder_hash: str) -> dict[str, Any]:
    folder = cache / split / info['name']
    marker = folder / 'manifest.json'
    sources = {'annotation': sha256_file(info['annotation']), 'builder': builder_hash}
    if split == 'train':
        sources['frame_labels'] = sha256_file(info['label'])
    if marker.is_file():
        previous = read_json(marker)
        if previous.get('sources') != sources or previous.get('status') != 'complete':
            raise RuntimeError(f'stale/incomplete CHAD video cache: {folder}')
        for name, digest in previous['array_sha256'].items():
            if not (folder / name).is_file() or sha256_file(folder / name) != digest:
                raise RuntimeError(f'changed cached array: {folder / name}')
        return previous
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f'partial video cache needs explicit recovery/new cache: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    samples = build_video_sequences(info)
    tracks = sorted({s.track_id for s in samples}, key=_pid_key)
    track_index = {track: index for index, track in enumerate(tracks)}
    spans, offset = ([], 0)
    for sample in samples:
        length = len(sample.frame_ids)
        spans.append({'start': offset, 'stop': offset + length, 'track_id': sample.track_id, 'segment_id': sample.segment_id, 'sample_id': sample.sample_id})
        offset += length

    def cat(field: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.concatenate([getattr(s, field).numpy() for s in samples], axis=0).astype(dtype) if samples else np.empty(shape, dtype=dtype)
    values = {'pose': cat('pose_features', (0, 85), np.float32), 'motion': cat('motion_features', (0, 31), np.float32), 'pose_valid': cat('pose_valid', (0,), bool), 'motion_valid': cat('motion_valid', (0,), bool), 'frame': cat('frame_ids', (0,), np.int32), 'track': np.concatenate([np.full(len(s.frame_ids), track_index[s.track_id], np.int32) for s in samples]) if samples else np.empty(0, np.int32)}
    if values['pose'].shape != (offset, 85) or values['motion'].shape != (offset, 31):
        raise RuntimeError('P85/M31 feature dimensions changed')
    if not np.isfinite(values['pose']).all() or not np.isfinite(values['motion']).all():
        raise RuntimeError('non-finite CHAD P85/M31 features')
    label_coverage = None
    if split == 'train':
        labels = np.load(info['label'], allow_pickle=False)
        if labels.shape != (int(info['frame_count']),) or not np.isin(labels, (0, 1)).all():
            raise RuntimeError('CHAD training frame labels are invalid')
        values['frame_label'] = labels[values['frame']].astype(np.uint8)
        covered = np.zeros(len(labels), dtype=bool)
        covered[values['frame'][values['pose_valid'] | values['motion_valid']]] = True
        label_coverage = {'positive_frame_count': int(labels.sum()), 'positive_frames_with_active_person': int((labels.astype(bool) & covered).sum()), 'positive_frames_without_active_person': int((labels.astype(bool) & ~covered).sum())}
    for name, value in values.items():
        np.save(folder / f'{name}.npy', value, allow_pickle=False)
    result = {'schema_version': CACHE_SCHEMA, 'status': 'complete', 'sources': sources, 'info': dict(info), 'row_count': offset, 'spans': spans, 'tracks': tracks, 'pose_dim': 85, 'motion_dim': 31, 'training_label_coverage': label_coverage, 'test_label_values_accessed': False, 'array_sha256': {f'{name}.npy': sha256_file(folder / f'{name}.npy') for name in values}}
    write_json_new(marker, result)
    return result

def build_sequence_cache(cache: Path, inventory: Mapping[str, Any], *, mode: str='formal', code_hashes: Mapping[str, str]=None) -> dict[str, Any]:
    """Per-video commit markers permit safe reuse after completed video stages."""
    code_hashes = code_hashes or {'features': sha256_file(Path(__file__).with_name('features.py')), 'motion': sha256_file(Path(__file__).with_name('motion.py')), 'orientation': sha256_file(Path(__file__).with_name('orientation.py')), 'data': sha256_file(Path(__file__).with_name('data.py')), 'quality': sha256_file(Path(__file__).with_name('quality.py')), 'chad_data': sha256_file(__file__)}
    identity = {'schema_version': CACHE_SCHEMA, 'mode': mode, 'inventory': inventory, 'code_hashes': dict(code_hashes), 'profile': FEATURE_PROFILE, 'quality_profile': 'legacy_v1'}
    digest = config_hash(identity)
    marker = cache / 'manifest.json'
    if marker.is_file() and read_json(marker).get('identity_sha256') != digest:
        raise RuntimeError('cache identity differs; use a separate cache directory')
    split_results: dict[str, Any] = {}
    for split in ('train', 'test'):
        videos, row_offset = ([], 0)
        for number, info in enumerate(inventory['splits'][split]['videos']):
            video = _video_cache(cache, info, split, config_hash(dict(code_hashes)))
            videos.append({**video, 'global_start': row_offset, 'global_stop': row_offset + video['row_count']})
            row_offset += video['row_count']
            print(f"[CHAD cache {split} {number + 1}] {info['name']}: {video['row_count']} rows", flush=True)
        split_results[split] = {'videos': videos, 'row_count': row_offset, 'frame_count': sum((int(v['info']['frame_count']) for v in videos))}
    result = {'schema_version': CACHE_SCHEMA, 'status': 'complete', 'identity_sha256': digest, 'mode': mode, 'splits': split_results, 'code_hashes': dict(code_hashes), 'test_label_values_accessed': False}
    if not marker.exists():
        write_json_new(marker, result)
    return result

def row_metadata(cache: Path, split_info: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    total = int(split_info['row_count'])
    fields = {'video': np.empty(total, np.int32), 'frame': np.empty(total, np.int32), 'track': np.empty(total, np.int32), 'active': np.empty(total, bool)}
    if split == 'train':
        fields['labels'] = np.empty(total, np.uint8)
    for index, video in enumerate(split_info['videos']):
        folder = cache / split / video['info']['name']
        section = slice(video['global_start'], video['global_stop'])
        fields['video'][section] = index
        for name in ('frame', 'track'):
            fields[name][section] = np.load(folder / f'{name}.npy', mmap_mode='r')
        fields['active'][section] = np.load(folder / 'pose_valid.npy', mmap_mode='r') | np.load(folder / 'motion_valid.npy', mmap_mode='r')
        if split == 'train':
            fields['labels'][section] = np.load(folder / 'frame_label.npy', mmap_mode='r')
    return fields

class ChadMotionDataset(Dataset):
    """One random crop per full segment, or fixed halo inference chunks."""

    def __init__(self, cache: Path, split_info: Mapping[str, Any], split: str, *, training: bool, max_frames: int, seed: int=20260815, halo_frames: int=32, labels: np.ndarray | None=None, weights: np.ndarray | None=None):
        self.cache, self.split_info, self.split = (Path(cache), split_info, split)
        self.training, self.max_frames, self.seed = (training, int(max_frames), int(seed))
        self.labels, self.weights, self.epoch = (labels, weights, 0)
        self.entries: list[tuple[int, Mapping[str, Any], int, int, int, int]] = []
        self.maps: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        if not training and (not 0 <= 2 * halo_frames < max_frames):
            raise ValueError('invalid halo/chunk sizes')
        for video_index, video in enumerate(split_info['videos']):
            for span in video['spans']:
                length = span['stop'] - span['start']
                if training:
                    self.entries.append((video_index, span, 0, length, 0, length))
                else:
                    core = max_frames - 2 * halo_frames
                    for start in range(0, length, core):
                        stop = min(length, start + core)
                        left, right = (max(0, start - halo_frames), min(length, stop + halo_frames))
                        self.entries.append((video_index, span, left, right, start - left, stop - left))

    def __len__(self) -> int:
        return len(self.entries)

    def _maps(self, index: int) -> dict[str, np.ndarray]:
        if index not in self.maps:
            folder = self.cache / self.split / self.split_info['videos'][index]['info']['name']
            self.maps[index] = {name: np.load(folder / f'{name}.npy', mmap_mode='r') for name in ARRAYS}
        self.maps.move_to_end(index)
        while len(self.maps) > 4:
            self.maps.popitem(last=False)
        return self.maps[index]

    def __getitem__(self, index: int) -> tuple[PersonSequenceSample, np.ndarray, int, int]:
        vi, span, left, right, keep_start, keep_stop = self.entries[index]
        video = self.split_info['videos'][vi]
        if self.training and right - left > self.max_frames:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + index * 97)
            left = int(torch.randint(right - left - self.max_frames + 1, (), generator=generator))
            right = left + self.max_frames
            keep_start, keep_stop = (0, right - left)
        start, stop = (span['start'] + left, span['start'] + right)
        arrays = self._maps(vi)
        rows = np.arange(video['global_start'] + start, video['global_start'] + stop, dtype=np.int64)
        tensor = lambda key: torch.from_numpy(np.array(arrays[key][start:stop], copy=True))
        label = torch.zeros(len(rows)) if self.labels is None else torch.from_numpy(self.labels[rows].astype(np.float32))
        weight = torch.zeros(len(rows)) if self.weights is None else torch.from_numpy(self.weights[rows].astype(np.float32))
        sample = PersonSequenceSample(pose_features=tensor('pose'), motion_features=tensor('motion'), scene_id=int(video['info']['camera']) - 1, labels=label, label_mask=weight > 0, label_weights=weight, pose_valid=tensor('pose_valid'), motion_valid=tensor('motion_valid'), frame_ids=tensor('frame').long(), sample_id=span['sample_id'], video=video['info']['canonical_video'], track_id=span['track_id'], segment_id=span['segment_id'])
        return (sample, rows, keep_start, keep_stop)

def collate_chad_motion(items: list[tuple[PersonSequenceSample, np.ndarray, int, int]]) -> dict[str, Any]:
    from .data import collate_person_sequences
    batch = collate_person_sequences([item[0] for item in items])
    batch['cache_rows'] = [item[1] for item in items]
    batch['keep_ranges'] = [(item[2], item[3]) for item in items]
    return batch

def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _chad_info(annotation_path, label_path, video_name, image_size):
    """Only load trusted official pickle files; pickle is not a safe interchange format."""
    with Path(annotation_path).open('rb') as stream:
        document = pickle.load(stream)
    if not isinstance(document, dict) or list(document) != list(range(len(document))):
        raise ValueError('CHAD frames must be contiguous, ordered and zero-based')
    camera = int(video_name.split('_')[0])
    if camera not in (1, 2, 3, 4):
        raise ValueError('CHAD camera must be 1, 2, 3 or 4')
    return {'name': video_name, 'camera': camera, 'canonical_video': f'chad_scene_{camera}_{video_name}', 'image_size': list(image_size), 'frame_count': len(document), 'row_count': sum((len(people) for people in document.values())), 'annotation': str(Path(annotation_path).resolve()), 'label': None if label_path is None else str(Path(label_path).resolve())}

def build_chad_inventory(root: str | Path) -> dict[str, Any]:
    """Build Split 2 metadata without reading any held-out label values."""
    root = Path(root)
    splits, seen = ({}, set())
    for split in ('train', 'test'):
        names = [line.strip() for line in (root / 'splits' / f'{split}_split_2.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if not names or len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError('invalid or overlapping CHAD splits')
        seen.update(names)
        videos, row_offset, frame_offset = ([], 0, 0)
        for name in names:
            camera = int(name.split('_')[0])
            size = (1280, 720) if camera == 4 else (1920, 1080)
            annotation, label = (root / 'annotations' / f'{name}.pkl', root / 'anomaly_labels' / f'{name}.npy')
            if not annotation.is_file() or not label.is_file():
                raise FileNotFoundError(f'missing CHAD files for {name}')
            info = _chad_info(annotation, label, name, size)
            info.update(index=len(videos), row_start=row_offset, row_stop=row_offset + info['row_count'], frame_start=frame_offset, frame_stop=frame_offset + info['frame_count'])
            row_offset += info['row_count']
            frame_offset += info['frame_count']
            videos.append(info)
        splits[split] = {'video_count': len(videos), 'row_count': row_offset, 'frame_count': frame_offset, 'videos': videos}
    return {'schema_version': 'chad-split2-inventory-v1', 'splits': splits, 'test_label_values_accessed': False}

def load_chad_video(annotation_path: str | Path, label_path: str | Path | None, video_name: str, image_size) -> tuple[tuple[PersonSequenceSample, ...], dict[str, np.ndarray]]:
    """Return label-blind sequences and row-aligned frame/track metadata.

    Optional frame labels belong to MIL bags, never to every person in a positive frame.
    """
    info = _chad_info(annotation_path, label_path, video_name, image_size)
    samples = build_video_sequences(info)
    tracks = {track: i for i, track in enumerate(sorted({s.track_id for s in samples}, key=_pid_key))}
    frame = np.concatenate([s.frame_ids.numpy() for s in samples]).astype(np.int32) if samples else np.empty(0, np.int32)
    rows = {'frame': frame, 'track': np.concatenate([np.full(len(s.frame_ids), tracks[s.track_id], np.int32) for s in samples]) if samples else np.empty(0, np.int32), 'active': np.concatenate([(s.pose_valid | s.motion_valid).numpy() for s in samples]) if samples else np.empty(0, bool)}
    if label_path is not None:
        labels = np.load(label_path, allow_pickle=False)
        if labels.shape != (info['frame_count'],) or not np.isin(labels, (0, 1)).all():
            raise ValueError('invalid CHAD frame labels')
        rows['labels'] = labels[frame].astype(np.uint8)
    return (samples, rows)
