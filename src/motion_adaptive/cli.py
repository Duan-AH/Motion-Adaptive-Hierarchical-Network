"""Command-line entry points for the two main experiments."""
from __future__ import annotations

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from .data import PersonSequenceSample, collate_person_sequences, load_pose_video, load_ub_split
from .evaluate import evaluate_scores, export_regions
from .inference import infer_video
from .learning import load_checkpoint, run_epoch, save_checkpoint, set_deterministic_seed, train_ubnormal
from .model import forward_batch, make_model
from .postprocess import postprocess_ubnormal


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + '\n', encoding='utf-8')


def configuration(dataset, path=None):
    raw = read_json(path) if path else json.loads(files('motion_adaptive').joinpath('config.json').read_text())
    return {**{key: value for key, value in raw.items() if key not in ('ubnormal', 'chad')}, **raw[dataset]}


def new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError('output is not empty; choose a new directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata(path):
    value = read_json(path)
    value = value.get('videos', value)
    if not value:
        raise ValueError('empty video inventory')
    for video, entry in value.items():
        if Path(video).name != video or '/' in video or '\\' in video:
            raise ValueError('video names must not contain paths')
        count = entry['frame_count']
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError('frame_count must be a positive integer')
    return value


def validate_labels(labels, inventory):
    if set(labels) != set(inventory):
        raise ValueError('label and full-video inventories differ')
    for video, values in labels.items():
        values = np.asarray(values)
        if values.shape != (inventory[video]['frame_count'],) or not np.isin(values, [0, 1]).all():
            raise ValueError(f'invalid full-video labels: {video}')


def self_check():
    """Small synthetic optimization, MIL, checkpoint and evaluation checks."""
    from .mil import build_mil_training_view, select_top1_positive_instances
    set_deterministic_seed(20260815)
    sample = PersonSequenceSample(
        pose_features=torch.randn(24, 85), motion_features=torch.rand(24, 31), scene_id=0,
        labels=torch.tensor([0., 1.] * 12), label_weights=torch.ones(24),
        frame_ids=torch.arange(24), video='synthetic', track_id='0',
    )
    batch = collate_person_sequences([sample])
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    result = run_epoch(model, [batch], device='cpu', optimizer=optimizer)
    labels = np.array([0, 0, 1, 1], np.uint8)
    active, video, frame, track = np.ones(4, bool), np.zeros(4, int), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    chosen, _ = select_top1_positive_instances(np.array([.1, .2, .3, .8]), labels, active, video, frame, track)
    view = build_mil_training_view(labels, active, video, frame, selected_positive=chosen)
    assert chosen.tolist() == [False, False, False, True] and view.fit_mask.tolist() == [True, True, False, True]
    with tempfile.TemporaryDirectory(prefix='motion-adaptive-check-') as folder:
        path = Path(folder) / 'model.pt'
        save_checkpoint(path, model, 1, 'ubnormal', configuration('ubnormal'))
        loaded, _ = load_checkpoint(path)
        model.eval(); loaded.eval()
        with torch.no_grad():
            assert torch.equal(forward_batch(model, batch).logits, forward_batch(loaded, batch).logits)
    metrics = evaluate_scores({'v': np.array([.1, .9])}, {'v': [0, 1]})
    try:
        evaluate_scores({'v': [.1, .9]}, {'v': [0, 1], 'missing': [0, 1]})
    except ValueError:
        pass
    else:
        raise RuntimeError('evaluation did not reject omitted videos')
    print(json.dumps({'status': 'passed', 'parameters': sum(p.numel() for p in model.parameters()),
                      'synthetic_loss': result.loss, 'metrics': metrics}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('self-check')
    for name in ('train-ubnormal', 'train-chad', 'predict-ubnormal', 'predict-chad'):
        command = commands.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--device', default='cuda')
        if name.startswith('train-'):
            command.add_argument('--config', type=Path)
        else:
            command.add_argument('--checkpoint', type=Path, required=True)
        if name.endswith('chad'):
            command.add_argument('--data-root', type=Path, required=True)
            command.add_argument('--cache', type=Path)
        elif name.startswith('train-'):
            for key in ('train', 'validation', 'train-metadata', 'validation-metadata', 'validation-labels'):
                command.add_argument('--' + key, type=Path, required=True)
        else:
            command.add_argument('--pose-dir', type=Path, required=True)
            command.add_argument('--metadata', type=Path, required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--scores', type=Path, required=True)
    labels = evaluate.add_mutually_exclusive_group(required=True)
    labels.add_argument('--labels', type=Path)
    labels.add_argument('--chad-root', type=Path)
    evaluate.add_argument('--output', type=Path, required=True)
    export = commands.add_parser('export-regions')
    export.add_argument('--predictions', type=Path, required=True)
    export.add_argument('--pose-dir', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.command == 'self-check':
        self_check()
        return 0
    if args.command == 'evaluate':
        if args.output.exists():
            raise FileExistsError(args.output)
        truth = read_json(args.labels) if args.labels else {
            name: np.load(args.chad_root / 'anomaly_labels' / (name + '.npy'), allow_pickle=False)
            for name in (args.chad_root / 'splits' / 'test_split_2.txt').read_text(encoding='utf-8-sig').split()}
        result = evaluate_scores(read_json(args.scores), truth)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps(result))
        return 0
    if args.command == 'export-regions':
        if not (args.predictions / 'COMPLETE.json').is_file():
            raise ValueError('prediction directory is incomplete')
        rows = [json.loads(line) for line in (args.predictions / 'person_scores.jsonl').read_text().splitlines() if line.strip()]
        result = export_regions(rows, read_json(args.predictions / 'frame_scores.json'), args.pose_dir, args.output)
        print(json.dumps(result))
        return 0
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device is unavailable; use --device cpu')
    output = new_output(args.output)
    dataset = args.command.split('-')[1]
    if args.command.startswith('train-'):
        config = configuration(dataset, args.config)
        write_json(output / 'config.json', config)
        if dataset == 'ubnormal':
            train_meta, val_meta = metadata(args.train_metadata), metadata(args.validation_metadata)
            if set(train_meta) & set(val_meta):
                raise ValueError('TRAIN and validation video inventories overlap')
            truth = read_json(args.validation_labels)
            validate_labels(truth, val_meta)
            train = load_ub_split(args.train, train_meta, expected_split='train')
            validation = load_ub_split(args.validation, val_meta, expected_split='validation')
            checkpoint = train_ubnormal(config, train, validation, truth, output, device)
        else:
            from .chad import train_chad
            from .chad_data import build_chad_inventory, build_sequence_cache
            cache = args.cache or output / 'cache'
            inventory = build_chad_inventory(args.data_root)
            manifest = build_sequence_cache(cache, inventory)
            checkpoint = train_chad(config, cache, manifest, output, device)
        write_json(output / 'COMPLETE.json', {'dataset': dataset, 'seed': config['seed'], 'checkpoint': checkpoint.name})
        print(str(checkpoint))
        return 0
    model, payload = load_checkpoint(args.checkpoint, device)
    if payload['dataset'] != dataset:
        raise ValueError('checkpoint and requested dataset differ')
    config = payload['config']
    if dataset == 'ubnormal':
        inventory = metadata(args.metadata)
        paths = {p.name.removesuffix('_alphapose_tracked_person.json'): p for p in args.pose_dir.glob('*_alphapose_tracked_person.json')}
        if set(paths) != set(inventory):
            raise ValueError('pose directory and full-video metadata inventories differ')
        scores = {}
        with (output / 'person_scores.jsonl').open('w', encoding='utf-8') as stream:
            for index, name in enumerate(sorted(inventory)):
                video = load_pose_video(paths[name], inventory[name])
                _, rows = infer_video(model, video, device, max_frames=int(config['eval_chunk_frames']),
                                      halo_frames=int(config['eval_halo_frames']), batch_size=int(config['batch_size']))
                final, processed = postprocess_ubnormal(rows, args.pose_dir, {name: video.frame_count})
                scores.update(final)
                for row in processed:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                print(f'UBnormal prediction {index + 1}/{len(inventory)}', flush=True)
    else:
        from .chad import predict_chad
        from .chad_data import build_chad_inventory, build_sequence_cache
        cache = args.cache or output / 'cache'
        manifest = build_sequence_cache(cache, build_chad_inventory(args.data_root))
        scores = predict_chad(model, cache, manifest, config, device)
    write_json(output / 'frame_scores.json', scores)
    write_json(output / 'COMPLETE.json', {'dataset': dataset, 'seed': config['seed'], 'videos': len(scores),
                                         'frames': sum(len(values) for values in scores.values()), 'test_labels_read': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
