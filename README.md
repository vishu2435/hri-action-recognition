# Ultra Vision

Computer vision final project for the project proposal's six KTH-style actions:
`walking`, `jogging`, `running`, `boxing`, `handwaving`, and `handclapping`.

The pipeline processes one 40-frame image folder at a time:

1. temporal foreground segmentation for a stationary camera;
2. single-person bounding-box tracking;
3. dense optical flow and kinematic feature extraction;
4. sequence-level Random Forest classification;
5. annotated video plus IoU, accuracy, per-class F1, and confusion-matrix evaluation.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install '.[dev]'
ultra-vision demo --fresh
streamlit run app.py
```

The demo command generates deterministic synthetic sequences, trains a model, evaluates a held-out split, and writes a sample annotated video under `artifacts/demo/`. The synthetic data is a smoke test, not a substitute for KTH validation.

## Use the provided dataset

Each sequence folder must contain at least 40 naturally named image files. Its annotation may be placed inside the folder as any `.txt` file:

```text
dataset/
  boxing/
    sequence_001/
      frame_001.png
      ...
      frame_040.png
      annotation.txt
```

The annotation format is the one specified in the proposal:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates may be normalized to 0-1 or expressed in pixels. Ultra Vision uses the following class map because the proposal defines IDs 1-6 but does not explicitly map each integer:

| ID | Action |
|---:|---|
| 1 | boxing |
| 2 | handclapping |
| 3 | handwaving |
| 4 | jogging |
| 5 | running |
| 6 | walking |

If the supplied annotations use a different map, edit `CLASS_NAMES` in `src/hri_action/constants.py` before training.

```bash
# Train
ultra-vision train /path/to/dataset --model artifacts/kth_model.joblib

# Classify one sequence and create an annotated MP4
ultra-vision predict /path/to/sequence --model artifacts/kth_model.joblib

# Evaluate labeled sequences
ultra-vision evaluate /path/to/test_dataset --model artifacts/kth_model.joblib
```

Evaluation writes `metrics.json` and `confusion_matrix.png`. Prediction writes `prediction.json`, `median_frame.png`, and `prediction.mp4`.

## Complete final-project experiment

The full experiment command enforces the proposal's dataset contract, creates a reproducible action-and-environment stratified split (48 train / 24 test), trains the classifier, and produces all required quantitative and qualitative outputs:

```bash
ultra-vision validate /path/to/dataset
ultra-vision experiment /path/to/dataset --output artifacts/kth_experiment
```

Expected environment markers `d1`, `d2`, `d3`, and `d4` may appear in any sequence path component. The experiment writes:

- `dataset_validation.json` — structural and annotation checks;
- `split_manifest.csv` — reproducible train/test membership;
- `action_model.joblib` — trained classifier and metadata;
- `evaluation/metrics.json` — accuracy, macro/per-class F1, precision, recall, and mIoU;
- `evaluation/predictions.csv` — sequence-level results;
- `evaluation/confusion_matrix.png` — required class-confusion analysis;
- `feature_importance.png` — interpretable motion-feature importance;
- `qualitative_samples/` — one annotated image and video per action.

For development with a smaller dataset, add `--allow-nonstandard`. The Streamlit **Train & evaluate** tab exposes the same complete workflow without the command line.

## Design scope

- Assumes one actor and a stationary camera, matching the proposal.
- Uses no neural action-recognition model; all classification inputs are interpretable motion and silhouette features.
- Distinguishes clap vs. wave with flow direction, upper-body motion, and inward/outward horizontal flow.
- Distinguishes jog vs. run with centroid velocity, flow magnitude, and dominant motion frequency.
- The segmenter is intentionally lightweight. MediaPipe can later be added as the proposal's fallback for difficult scale/clothing conditions.

## Tests

```bash
pytest -q
```
