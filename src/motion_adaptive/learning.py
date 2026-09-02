"""Shared optimization and the single-seed UBnormal training chain."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .model import forward_batch, make_model
from .evaluate import binary_auroc


@dataclass(frozen=True)
class EpochResult:
    loss: float
    labeled_count: int
    frame_auroc: float | None = None


def set_deterministic_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def weighted_classification_loss(logits: Tensor, targets: Tensor, active: Tensor,
                                 *, label_weights: Tensor | None,
                                 pos_weight: float) -> tuple[Tensor, int]:
    active = active.bool()
    count = int(active.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    element = F.binary_cross_entropy_with_logits(
        logits[active], targets[active].to(logits.dtype), reduction="none",
        pos_weight=torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )
    if label_weights is not None:
        weights = label_weights[active].to(logits.dtype).clamp_min(0)
        if float(weights.sum()) <= 0:
            return logits.sum() * 0.0, 0
        return (element * weights).sum() / weights.sum(), count
    return element.mean(), count


def run_epoch(model: nn.Module, loader: Any, *, device: str | torch.device,
              pos_weight: float = 1.0, optimizer: torch.optim.Optimizer | None = None,
              max_steps: int | None = None, gradient_clip: float = 5.0,
              frame_labels: Mapping[str, Any] | None = None) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    device = torch.device(device)
    total_loss = 0.0
    total_count = 0
    frame_scores = {str(video): np.zeros(len(labels), dtype=np.float64)
                    for video, labels in frame_labels.items()} if frame_labels is not None else None
    emitted: set[tuple[str, str, int]] = set()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, raw in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = {key: value.to(device) if isinstance(value, Tensor) else value
                     for key, value in raw.items()}
            output = forward_batch(model, batch)
            if not torch.isfinite(output.logits).all() or not torch.isfinite(output.probabilities).all():
                raise RuntimeError(f"non-finite model output at batch {step}")
            if frame_scores is not None:
                probabilities = output.probabilities.detach().cpu()
                prediction_mask = output.prediction_mask.detach().cpu().bool()
                sequence_mask = batch["sequence_mask"].detach().cpu().bool()
                frame_ids = batch["frame_ids"].detach().cpu()
                for index, (video, track_id) in enumerate(zip(batch["videos"], batch["track_ids"])):
                    if video not in frame_scores:
                        raise ValueError(f"undeclared validation video: {video}")
                    for position in sequence_mask[index].nonzero(as_tuple=False).flatten().tolist():
                        frame = int(frame_ids[index, position])
                        if not 0 <= frame < len(frame_scores[video]):
                            raise ValueError(f"validation frame out of range: {video}/{frame}")
                        identity = (str(video), str(track_id), frame)
                        if identity in emitted:
                            raise ValueError(f"duplicate validation person frame: {identity}")
                        emitted.add(identity)
                        if bool(prediction_mask[index, position]):
                            frame_scores[video][frame] = max(frame_scores[video][frame], float(probabilities[index, position]))
            active = batch["label_mask"].bool() & output.prediction_mask.bool()
            loss, count = weighted_classification_loss(
                output.logits, batch["labels"], active,
                label_weights=batch.get("label_weights"), pos_weight=pos_weight,
            )
            auxiliary, _ = model.auxiliary_training_loss(output, active)
            # Preserve the arithmetic/gradient graph of the trained objective.
            zero = output.logits.sum() * 0.0
            objective = loss + (zero + auxiliary) + zero
            if not torch.isfinite(objective):
                raise RuntimeError(f"non-finite training objective at batch {step}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            if count:
                total_loss += float(objective.detach()) * count
                total_count += count
    if not total_count:
        raise RuntimeError("epoch had no trusted labeled predictions")
    frame_auroc = None
    if frame_scores is not None:
        labels, scores = [], []
        for video in sorted(frame_scores):
            labels.extend(np.asarray(frame_labels[video], dtype=np.int8).tolist())
            scores.extend(frame_scores[video].tolist())
        frame_auroc = binary_auroc(labels, scores)
    return EpochResult(total_loss / total_count, total_count, frame_auroc)


def save_checkpoint(path: str | Path, model: nn.Module, epoch: int,
                    dataset: str, config: Mapping[str, Any]) -> None:
    """Save only plain metadata and tensors; no executable pickle objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "motion-adaptive-checkpoint-v1", "dataset": str(dataset),
        "epoch": int(epoch), "seed": int(config["seed"]),
        "model_config": asdict(model.config), "model_state": model.state_dict(),
        "config": json.loads(json.dumps(dict(config))),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != "motion-adaptive-checkpoint-v1":
        raise ValueError("unsupported checkpoint schema")
    model = make_model(payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, payload


def train_ubnormal(config: Mapping[str, Any], train_samples: Sequence[Any],
                   validation_samples: Sequence[Any], validation_labels: Mapping[str, Any],
                   output: Path, device: str | torch.device) -> Path:
    from .data import SequenceWindowDataset, collate_person_sequences, positive_class_weight
    from .motion import fit_motion_calibration

    output = Path(output)
    if any((output / name).exists() for name in ("best.pt", "last.pt", "history.json")):
        raise FileExistsError("training output already contains a run; choose a new directory")
    if {sample.video for sample in train_samples} & {sample.video for sample in validation_samples}:
        raise ValueError("train/validation videos overlap")
    seed = int(config["seed"])
    set_deterministic_seed(seed)
    evidence, video_ids, track_ids = [], [], []
    for sample in train_samples:
        valid = sample.motion_valid.bool()
        evidence.append(sample.motion_features[valid, 17:29])
        video_ids.extend([sample.video] * int(valid.sum()))
        track_ids.extend([sample.track_id] * int(valid.sum()))
    if not video_ids:
        raise ValueError("no valid training motion evidence")
    fitted = fit_motion_calibration(torch.cat(evidence), video_ids=video_ids, track_ids=track_ids)
    model_config = {**dict(config.get("model", {})), **fitted}
    train_dataset = SequenceWindowDataset(train_samples, max_frames=int(config["train_crop_frames"]), mode="train", seed=seed)
    validation_dataset = SequenceWindowDataset(validation_samples, max_frames=int(config["validation_chunk_frames"]), mode="eval")
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True,
                              generator=torch.Generator().manual_seed(seed),
                              collate_fn=collate_person_sequences, num_workers=0)
    # No explicit generator here: preserve the training chain's global RNG use.
    validation_loader = DataLoader(validation_dataset, batch_size=int(config["batch_size"]),
                                   shuffle=False, collate_fn=collate_person_sequences, num_workers=0)
    model = make_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    pos_weight = positive_class_weight(train_samples)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(fitted, indent=2) + "\n", encoding="utf-8")
    resolved_config = {**dict(config), "model": asdict(model.config)}
    (output / "config.json").write_text(json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8")
    best = float("-inf")
    history = []
    for epoch in range(int(config["epochs"])):
        train_dataset.set_epoch(epoch)
        training = run_epoch(model, train_loader, device=device, pos_weight=pos_weight,
                              optimizer=optimizer, gradient_clip=float(config["gradient_clip"]))
        validation = run_epoch(model, validation_loader, device=device, pos_weight=pos_weight,
                                frame_labels=validation_labels)
        if validation.frame_auroc is None:
            raise RuntimeError("validation omitted complete-video frame AUROC")
        is_best = validation.frame_auroc > best
        best = max(best, validation.frame_auroc)
        save_checkpoint(output / "last.pt", model, epoch, "ubnormal", resolved_config)
        if is_best:
            save_checkpoint(output / "best.pt", model, epoch, "ubnormal", resolved_config)
        history.append({"epoch": epoch, "train": asdict(training), "validation": asdict(validation)})
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"epoch": epoch + 1, "epochs": config["epochs"],
                          "train_loss": training.loss, "validation_frame_auroc": validation.frame_auroc,
                          "best": best}), flush=True)
    return output / "best.pt"
