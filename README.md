# Motion-Adaptive Hierarchical Network

Minimal main-experiment code for **Motion-Adaptive Hierarchical Network for
Supervised Skeleton-Based Video Anomaly Detection**.

Only the final model and its UBnormal / CHAD training, prediction and evaluation
paths are included. No datasets, derived person labels, pose estimators,
trackers, checkpoints, results, figures, baselines or ablation runners are
distributed.

## Install

Use Python 3.10 or newer. Install a PyTorch build appropriate for your CUDA
environment, then run from this repository:

```bash
python -m pip install -e .
motion-adaptive self-check
```

Tested with Python 3.10.20, PyTorch 2.7.1+cu128 and NumPy 1.26.4.
The self-check uses synthetic data and temporary files; it does not download
datasets or reproduce a benchmark result. Commands below accept `--device cpu`
instead of the default `cuda`. Use a new, empty output directory for each run.

## Model and fixed settings

The network encodes body-relative pose, motion and shape in three temporal
branches. Multi-span motion and orientation measurements guide a learned
three-expert router; hierarchical anomaly heads produce person-frame scores.
The three experts have no prescribed low/medium/high-motion labels.

`src/motion_adaptive/config.json` is loaded by training, not just documentation.
It specifies the single seed **20260815**, 128-frame crops, batch size 16,
AdamW (learning rate 0.0003, weight decay 0.0001), and the final routing loss.
Motion normalization is fitted only on training data and saved in checkpoints.

- UBnormal: 30 epochs; select the earliest best validation frame AUROC using
  non-overlapping 512-frame chunks. TEST uses 256-frame chunks with 32-frame
  context halos, person maxima, fixed gap filling, then frame despiking.
- CHAD: official split 2; 12 initialization epochs followed by three six-epoch
  top-1 frame-bag MIL phases, keeping the same optimizer. Use the final model;
  TEST uses 512-frame chunks with 32-frame halos and raw person maxima.

## UBnormal inputs

Acquire the dataset and tracked COCO-17 poses separately. Frame indices are
zero-based; coordinates are pixels. Each TEST pose file is named
`VIDEO_alphapose_tracked_person.json` and contains a dictionary of track IDs,
then frame IDs, then an object with `keypoints`: 51 flattened `(x,y,confidence)`
values. Empty videos still need an empty `{}` pose file and metadata entry.

Each split has a complete metadata JSON, including videos without valid poses:

```json
{"normal_scene_1_scenario1_1": {"frame_count": 451, "width": 1280, "height": 720}}
```

Training and validation require **prepared person-level matching labels** in
`MATCHING_DIR/videos/VIDEO.jsonl`, one object per observed person-frame:

```text
video, split, predicted_track_id, segment_id, frame_index, keypoints,
label, confidence, confidence_tier, suspected_split
```

`split` is `train` or `validation`; `label` is `normal`, `abnormal`, or
`unknown`; `confidence` is a weight in [0,1]. The fixed training filter retains
known labels with tier `anchor` or `propagated` and `suspected_split=false`.
There must be exactly one matching file per declared video. Matching-label
construction and manual review are outside this minimal release; binary frame
labels alone do **not** substitute for these person labels.

Validation and TEST frame-label JSON files map every video to its full vector
of **0 = normal, 1 = abnormal**. If using UBnormal `*_tracks.txt` NumPy arrays
whose convention is 1 = normal, invert them once when preparing this JSON.
Do not invert labels already in the required convention.

```bash
motion-adaptive train-ubnormal --train /path/train_matching --validation /path/validation_matching --train-metadata /path/train_metadata.json --validation-metadata /path/validation_metadata.json --validation-labels /path/validation_labels.json --output runs/ubnormal
motion-adaptive predict-ubnormal --checkpoint runs/ubnormal/best.pt --pose-dir /path/test_poses --metadata /path/test_metadata.json --output runs/ubnormal_test
motion-adaptive evaluate --scores runs/ubnormal_test/frame_scores.json --labels /path/test_labels.json --output runs/ubnormal_metrics.json
```

Predictions cover every declared frame. Raw scores are zero when no valid
person is present; UBnormal gap filling can extend evidence into missing
observations. Evaluation rejects missing or extra videos, invalid labels and
incomplete frame vectors.

## CHAD inputs and commands

Use the official release layout:

```text
CHAD/
  annotations/VIDEO.pkl
  anomaly_labels/VIDEO.npy
  splits/train_split_2.txt
  splits/test_split_2.txt
```

Only load trusted dataset pickle files. Labels use 0 = normal, 1 = abnormal.
The code preserves official split order and builds a pose-feature cache without
reading TEST label values. Use a new cache for this release, then reuse it for
prediction; caches from other implementations are not compatible.

```bash
motion-adaptive train-chad --data-root /path/CHAD --cache runs/chad_cache --output runs/chad
motion-adaptive predict-chad --checkpoint runs/chad/model.pt --data-root /path/CHAD --cache runs/chad_cache --output runs/chad_test
motion-adaptive evaluate --scores runs/chad_test/frame_scores.json --chad-root /path/CHAD --output runs/chad_metrics.json
```

Prediction writes `frame_scores.json` and a completion marker before evaluation
reads TEST labels. All official TEST videos and frames enter the reported
micro frame AUROC and average precision.

## UBnormal region/track evaluation

Export the prediction format for the external AED evaluator:

```bash
motion-adaptive export-regions --predictions runs/ubnormal_test --pose-dir /path/test_poses --output runs/ubnormal_regions
```

Each exported row contains `frame x1 y1 x2 y2 score`. Boxes come from the
tracked poses; the network does not predict bounding boxes. Use the original
[AED evaluation code](https://github.com/lilygeorgescu/AED/tree/master/evaluation)
and its UBnormal ground truth for RBDC/TBDC, following that project's license
and evaluation instructions. The evaluator and ground truth are not vendored.

## Reproducibility boundary

The release is checked against the original final implementation for features,
model outputs, gradients, optimizer updates, CHAD MIL and inference, and the
full 211-video UBnormal postprocessing / region-export path. Small synthetic
training and checkpoint tests check the public entry points. A complete
benchmark retraining is not part of this release check. Exact training results
also depend on the same prepared input annotations and numerical environment.

## License

MIT. Dataset and external evaluator licenses remain separate.
