from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import CLASS_NAMES, FRAME_COUNT, IMAGE_EXTENSIONS, NAME_TO_ID


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


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
        return f"d{match.group(1)}" if match else "unknown"


def parse_annotation(path: Path) -> Annotation:
    values = path.read_text(encoding="utf-8").strip().split()
    if not values:
        raise ValueError(f"Empty annotation: {path}")
    class_id = int(float(values[0]))
    if class_id not in CLASS_NAMES:
        raise ValueError(f"Class id must be in 1..6: {path}")
    bbox = tuple(float(v) for v in values[1:5]) if len(values) >= 5 else None
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


def discover_sequences(root: str | Path, frame_count: int = FRAME_COUNT) -> list[SequenceRecord]:
    root = Path(root)
    records: list[SequenceRecord] = []
    for folder in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        frames = sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=natural_key,
        )
        if len(frames) >= frame_count:
            records.append(SequenceRecord(folder, tuple(frames[:frame_count]), _annotation_for(folder), len(frames)))
    return records


def load_frames(record_or_path: SequenceRecord | str | Path, frame_count: int = FRAME_COUNT) -> list[np.ndarray]:
    if isinstance(record_or_path, SequenceRecord):
        paths = record_or_path.frame_paths
    else:
        folder = Path(record_or_path)
        paths = tuple(sorted(
            (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key
        )[:frame_count])
    if len(paths) != frame_count:
        raise ValueError(f"Expected exactly {frame_count} frames, found {len(paths)}")
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(frame is None for frame in frames):
        raise ValueError(f"Could not decode every frame in {Path(paths[0]).parent}")
    height, width = frames[0].shape[:2]
    return [cv2.resize(frame, (width, height)) if frame.shape[:2] != (height, width) else frame for frame in frames]


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
