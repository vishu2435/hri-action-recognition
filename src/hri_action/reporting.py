from __future__ import annotations

from pathlib import Path

import cv2

from .model import Prediction
from .vision import annotate_frame


def write_annotated_video(
    frames: list, prediction: Prediction, path: str | Path, fps: float = 25.0
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video at {path}")
    for frame, box in zip(frames, prediction.track.boxes):
        writer.write(annotate_frame(frame, box, prediction.label, prediction.confidence))
    writer.release()
    return path


def write_preview_frame(frames: list, prediction: Prediction, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = annotate_frame(frames[19], prediction.track.boxes[19], prediction.label, prediction.confidence)
    cv2.imwrite(str(path), rendered)
    return path

