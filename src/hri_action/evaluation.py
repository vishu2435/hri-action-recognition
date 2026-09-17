from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)

from .constants import CLASS_NAMES
from .dataset import SequenceRecord, load_frames, normalized_bbox_to_pixels
from .model import ActionRecognizer
from .vision import bbox_iou


def evaluate(model: ActionRecognizer, records: list[SequenceRecord], output_dir: str | Path) -> dict[str, object]:
    usable = [record for record in records if record.annotation is not None]
    if not usable:
        raise ValueError("No labeled sequences found")
    truth: list[int] = []
    predicted: list[int] = []
    ious: list[float] = []
    rows: list[dict[str, object]] = []
    for record in usable:
        frames = load_frames(record)
        prediction = model.predict(frames)
        annotation = record.annotation
        assert annotation is not None
        truth.append(annotation.class_id)
        predicted.append(prediction.class_id)
        iou: float | None = None
        if annotation.bbox_xywh is not None:
            h, w = frames[19].shape[:2]
            ground_truth = normalized_bbox_to_pixels(annotation.bbox_xywh, w, h)
            iou = bbox_iou(prediction.track.boxes[19], ground_truth)
            ious.append(iou)
        rows.append({"sequence": record.name, "truth": CLASS_NAMES[annotation.class_id],
                     "scenario": record.scenario, "prediction": prediction.label,
                     "confidence": prediction.confidence, "iou": iou})

    labels = sorted(CLASS_NAMES)
    metrics: dict[str, object] = {
        "sequences": len(usable),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {CLASS_NAMES[class_id]: float(score) for class_id, score in zip(
            labels, f1_score(truth, predicted, labels=labels, average=None, zero_division=0)
        )},
        "per_class_precision": {CLASS_NAMES[class_id]: float(score) for class_id, score in zip(
            labels, precision_score(truth, predicted, labels=labels, average=None, zero_division=0)
        )},
        "per_class_recall": {CLASS_NAMES[class_id]: float(score) for class_id, score in zip(
            labels, recall_score(truth, predicted, labels=labels, average=None, zero_division=0)
        )},
        "mean_iou": float(np.mean(ious)) if ious else None,
        "results": rows,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sequence", "truth", "scenario", "prediction", "confidence", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=[CLASS_NAMES[i] for i in labels]).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35
    )
    ax.set_title("Sequence classification confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=170)
    plt.close(fig)
    return metrics
