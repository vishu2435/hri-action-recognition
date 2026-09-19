# Author: Himashi
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import CLASS_NAMES, FRAME_COUNT, IMAGE_EXTENSIONS, NAME_TO_ID


def natural_key(path: Path) -> list[object]:
    key_parts = []
    name_parts = re.split(r"(\d+)", path.name)

    for part in name_parts:
        if part.isdigit():
            key_parts.append(int(part))
        else:
            key_parts.append(part.lower())

    return key_parts


@dataclass(frozen=True)
class Annotation:
    class_id: int
    bbox_xywh: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SequenceRecord:
    path: Path
    frame_paths: tuple[Path, ...]
    annotation: Annotation | None
    total_frame_count: int = FRAME_COUNT

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def scenario(self) -> str:
        """Infer KTH scenario d1..d4 from any path component."""
        match = re.search(r"(?:^|[^a-z0-9])d([1-4])(?:[^a-z0-9]|$)", str(self.path).lower())

        if match:
            return f"d{match.group(1)}"

        return "unknown"


def parse_annotation(path: Path) -> Annotation:
    values = path.read_text(encoding="utf-8").strip().split()
    if not values:
        raise ValueError(f"Empty annotation: {path}")
    class_id = int(float(values[0]))

    if class_id not in CLASS_NAMES:
        raise ValueError(f"Class id must be in 1..6: {path}")

    bbox = None
    if len(values) >= 5:
        bbox_values = []

        for value in values[1:5]:
            bbox_values.append(float(value))

        bbox = tuple(bbox_values)

    return Annotation(class_id, bbox)  # type: ignore[arg-type]


def _annotation_for(folder: Path) -> Annotation | None:
    candidates = sorted(folder.glob("*.txt"))
    sibling = folder.with_suffix(".txt")

    if sibling.exists():
        candidates.append(sibling)

    for candidate in candidates:
        try:
            return parse_annotation(candidate)
        except (ValueError, OSError):
            continue

    lower_path = str(folder).lower()

    for name, class_id in NAME_TO_ID.items():
        if name in lower_path:
            return Annotation(class_id)

    return None


def _all_folders(root: Path) -> list[Path]:
    folders = [root]

    child_folders = []
    for path in root.rglob("*"):
        if path.is_dir():
            child_folders.append(path)

    for folder in sorted(child_folders):
        folders.append(folder)

    return folders


def _image_files(folder: Path) -> list[Path]:
    frames = []

    for path in folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            frames.append(path)

    return sorted(frames, key=natural_key)


def discover_sequences(root: str | Path, frame_count: int = FRAME_COUNT) -> list[SequenceRecord]:
    root = Path(root)
    records: list[SequenceRecord] = []

    for folder in _all_folders(root):
        frames = _image_files(folder)

        if len(frames) >= frame_count:
            selected_frames = tuple(frames[:frame_count])
            annotation = _annotation_for(folder)
            record = SequenceRecord(folder, selected_frames, annotation, len(frames))
            records.append(record)

    return records


def load_frames(record_or_path: SequenceRecord | str | Path, frame_count: int = FRAME_COUNT) -> list[np.ndarray]:
    if isinstance(record_or_path, SequenceRecord):
        paths = record_or_path.frame_paths
    else:
        folder = Path(record_or_path)
        image_paths = _image_files(folder)
        paths = tuple(image_paths[:frame_count])

    if len(paths) != frame_count:
        raise ValueError(f"Expected exactly {frame_count} frames, found {len(paths)}")

    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        frames.append(frame)

    for frame in frames:
        if frame is None:
            raise ValueError(f"Could not decode every frame in {Path(paths[0]).parent}")

    if not frames:
        raise ValueError(f"Could not decode every frame in {Path(paths[0]).parent}")

    height, width = frames[0].shape[:2]

    resized_frames = []
    for frame in frames:
        if frame.shape[:2] != (height, width):
            resized_frame = cv2.resize(frame, (width, height))
            resized_frames.append(resized_frame)
        else:
            resized_frames.append(frame)

    return resized_frames


def normalized_bbox_to_pixels(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    xc, yc, bw, bh = bbox

    if max(abs(xc), abs(yc), abs(bw), abs(bh)) <= 1.5:
        xc, bw = xc * width, bw * width
        yc, bh = yc * height, bh * height

    return (
        int(round(xc - bw / 2)), int(round(yc - bh / 2)),
        int(round(bw)), int(round(bh)),
    )
