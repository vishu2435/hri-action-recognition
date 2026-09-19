# Author: Vishesh
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import CLASS_NAMES
from .dataset import SequenceRecord, discover_sequences, load_frames
from .evaluation import evaluate
from .features import FEATURE_NAMES, extract_features
from .model import ActionRecognizer
from .reporting import write_annotated_video, write_preview_frame
from .validation import DatasetValidation, validate_dataset


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    model_path: Path
    metrics: dict[str, object]
    validation: DatasetValidation
    train_count: int
    test_count: int


def stratified_split(
    records: list[SequenceRecord], seed: int = 42, test_per_stratum: int = 1
) -> tuple[list[SequenceRecord], list[SequenceRecord]]:
    """Split by action and scenario; for the proposal this yields 48 train / 24 test."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, str], list[SequenceRecord]] = defaultdict(list)
    for record in records:
        if record.annotation is not None:
            groups[(record.annotation.class_id, record.scenario)].append(record)
    train: list[SequenceRecord] = []
    test: list[SequenceRecord] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: str(record.path))
        order = rng.permutation(len(group))
        shuffled = [group[index] for index in order]
        holdout = min(test_per_stratum, max(0, len(shuffled) - 1))
        test.extend(shuffled[:holdout])
        train.extend(shuffled[holdout:])
    return sorted(train, key=lambda r: str(r.path)), sorted(test, key=lambda r: str(r.path))


def _extract_matrix(records: list[SequenceRecord], cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    expected_names = np.asarray([str(record.path) for record in records])
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cached["paths"], expected_names):
            return cached["features"], cached["labels"]
    features: list[np.ndarray] = []
    labels: list[int] = []
    for index, record in enumerate(records, start=1):
        vector, _ = extract_features(load_frames(record))
        features.append(vector)
        assert record.annotation is not None
        labels.append(record.annotation.class_id)
        print(f"[{index:>3}/{len(records)}] extracted {record.name}")
    matrix = np.vstack(features)
    targets = np.asarray(labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, paths=expected_names, features=matrix, labels=targets)
    return matrix, targets


def _write_manifest(path: Path, train: list[SequenceRecord], test: list[SequenceRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "sequence", "class_id", "class_name", "scenario"])
        writer.writeheader()
        for split, records in (("train", train), ("test", test)):
            for record in records:
                assert record.annotation is not None
                writer.writerow({"split": split, "sequence": str(record.path),
                                 "class_id": record.annotation.class_id,
                                 "class_name": CLASS_NAMES[record.annotation.class_id],
                                 "scenario": record.scenario})


def _write_feature_importance(model: ActionRecognizer, path: Path) -> None:
    importance = model.estimator.feature_importances_
    order = np.argsort(importance)[-15:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.asarray(FEATURE_NAMES)[order], importance[order], color="#39dea6")
    ax.set_title("Top motion-feature importances")
    ax.set_xlabel("Random Forest importance")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_experiment(
    dataset: str | Path, output_dir: str | Path, seed: int = 42,
    strict_proposal: bool = True, qualitative_videos: bool = True,
) -> ExperimentResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_sequences(dataset)
    validation = validate_dataset(records, strict_proposal=strict_proposal)
    (output_dir / "dataset_validation.json").write_text(
        json.dumps(validation.as_dict(), indent=2), encoding="utf-8"
    )
    if validation.errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(validation.errors))
    train, test = stratified_split(records, seed=seed)
    if not train or not test:
        raise ValueError("Dataset cannot be split into non-empty train and test sets")
    _write_manifest(output_dir / "split_manifest.csv", train, test)
    train_features, train_labels = _extract_matrix(train, output_dir / "train_features.npz")
    model = ActionRecognizer(metadata={
        "dataset": str(Path(dataset).resolve()), "seed": seed,
        "train_sequences": len(train), "test_sequences": len(test),
        "strict_proposal": strict_proposal,
    }).fit_features(train_features, train_labels)
    model_path = output_dir / "action_model.joblib"
    model.save(model_path)
    metrics = evaluate(model, test, output_dir / "evaluation")
    _write_feature_importance(model, output_dir / "feature_importance.png")

    if qualitative_videos:
        seen: set[int] = set()
        samples_dir = output_dir / "qualitative_samples"
        for record in test:
            assert record.annotation is not None
            if record.annotation.class_id in seen:
                continue
            seen.add(record.annotation.class_id)
            frames = load_frames(record)
            prediction = model.predict(frames)
            stem = CLASS_NAMES[record.annotation.class_id]
            write_preview_frame(frames, prediction, samples_dir / f"{stem}.png")
            write_annotated_video(frames, prediction, samples_dir / f"{stem}.mp4")

    summary = {
        "proposal": "Ultra Vision",
        "split": {"train": len(train), "test": len(test), "seed": seed},
        "metrics": {key: value for key, value in metrics.items() if key != "results"},
        "artifacts": {
            "model": str(model_path), "manifest": str(output_dir / "split_manifest.csv"),
            "confusion_matrix": str(output_dir / "evaluation" / "confusion_matrix.png"),
            "feature_importance": str(output_dir / "feature_importance.png"),
        },
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ExperimentResult(output_dir, model_path, metrics, validation, len(train), len(test))

