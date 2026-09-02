"""Fixed CHAD split-2 training and full-video prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .chad_data import ChadMotionDataset, collate_chad_motion, row_metadata
from .learning import run_epoch, save_checkpoint, set_deterministic_seed
from .mil import build_mil_training_view, select_top1_positive_instances
from .model import forward_batch, make_model
from .motion import fit_motion_calibration


def loader_for(dataset, config, epoch=0):
    dataset.epoch = epoch
    return DataLoader(
        dataset, batch_size=int(config['batch_size']), shuffle=dataset.training,
        generator=torch.Generator().manual_seed(int(config['seed']) + epoch * 1009),
        collate_fn=collate_chad_motion, num_workers=0,
    )


def score_rows(model, cache, split_info, split, config, device):
    dataset = ChadMotionDataset(
        cache, split_info, split, training=False,
        max_frames=int(config['eval_chunk_frames']),
        halo_frames=int(config['eval_halo_frames']), seed=int(config['seed']),
    )
    total = int(split_info['row_count'])
    scores, active, emitted = np.zeros(total, np.float32), np.zeros(total, bool), np.zeros(total, bool)
    model.eval()
    with torch.no_grad():
        for batch in loader_for(dataset, config):
            moved = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = forward_batch(model, moved)
            probabilities = torch.sigmoid(output.logits).cpu().numpy()
            valid = output.prediction_mask.cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError('non-finite person scores')
            for index, (rows, (left, right)) in enumerate(zip(batch['cache_rows'], batch['keep_ranges'])):
                keep = rows[left:right]
                if emitted[keep].any():
                    raise RuntimeError('duplicate person frame in halo inference')
                scores[keep] = np.where(valid[index, left:right], probabilities[index, left:right], 0.0)
                active[keep] = valid[index, left:right]
                emitted[keep] = True
    if not emitted.all():
        raise RuntimeError('halo inference omitted person frames')
    return scores, active


def _calibration(cache, train_info):
    evidence, videos, tracks = [], [], []
    for video in train_info['videos']:
        name = video['info']['name']
        folder = Path(cache) / 'train' / name
        motion = np.load(folder / 'motion.npy', mmap_mode='r')
        valid = np.load(folder / 'motion_valid.npy', mmap_mode='r').astype(bool)
        track = np.load(folder / 'track.npy', mmap_mode='r')
        evidence.append(np.asarray(motion[valid, 17:29], np.float64))
        videos.extend([name] * int(valid.sum()))
        tracks.extend([str(int(value)) for value in track[valid]])
    return fit_motion_calibration(torch.from_numpy(np.concatenate(evidence)), video_ids=videos, track_ids=tracks)


def train_chad(config, cache, cache_manifest, output, device):
    """Twelve initialization epochs, then three six-epoch top-1 MIL phases."""
    output = Path(output)
    set_deterministic_seed(int(config['seed']))
    profile = _calibration(cache, cache_manifest['splits']['train'])
    model = make_model({**config['model'], **profile}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']),
                                 weight_decay=float(config['weight_decay']))
    meta = row_metadata(cache, cache_manifest['splits']['train'], 'train')
    epoch, history = 0, []
    for phase, count in enumerate(config['phase_epochs']):
        selected = None
        if phase:
            scores, active = score_rows(model, cache, cache_manifest['splits']['train'], 'train', config, device)
            if not np.array_equal(active, meta['active']):
                raise RuntimeError('MIL input and model validity disagree')
            selected, _ = select_top1_positive_instances(
                scores, meta['labels'], meta['active'], meta['video'], meta['frame'], meta['track'])
        view = build_mil_training_view(meta['labels'], meta['active'], meta['video'], meta['frame'],
                                       selected_positive=selected)
        dataset = ChadMotionDataset(
            cache, cache_manifest['splits']['train'], 'train', training=True,
            max_frames=int(config['train_crop_frames']), seed=int(config['seed']),
            labels=view.labels, weights=view.weights,
        )
        for _ in range(int(count)):
            result = run_epoch(model, loader_for(dataset, config, epoch), device=device,
                               pos_weight=1.0, optimizer=optimizer, gradient_clip=float(config['gradient_clip']))
            epoch += 1
            history.append({'epoch': epoch, 'phase': phase, 'loss': result.loss})
            print(f'CHAD epoch {epoch}: loss={result.loss:.6f}', flush=True)
    checkpoint = output / 'model.pt'
    save_checkpoint(checkpoint, model=model, epoch=epoch, dataset='chad', config=config)
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return checkpoint


def predict_chad(model, cache, cache_manifest, config, device):
    info = cache_manifest['splits']['test']
    scores, active = score_rows(model, cache, info, 'test', config, device)
    meta = row_metadata(cache, info, 'test')
    if not np.array_equal(active, meta['active']):
        raise RuntimeError('TEST input and model validity disagree')
    frames = {}
    for video in info['videos']:
        section = slice(video['global_start'], video['global_stop'])
        values = np.zeros(int(video['info']['frame_count']), np.float32)
        np.maximum.at(values, meta['frame'][section], scores[section])
        frames[video['info']['name']] = values
    return frames
