from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .constants import CLASS_NAMES


def _person_frame(action: str, t: int, rng: np.random.Generator, variant: int) -> np.ndarray:
    height, width = 144, 192
    scenario = variant % 4
    if scenario == 0:
        base = 182
    elif scenario == 1:
        base = 155 + int(t * .4)
    elif scenario == 2:
        base = 205
    else:
        base = 116
    frame = np.full((height, width, 3), base, dtype=np.uint8)
    noise = rng.normal(0, 3.0, frame.shape[:2]).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)
    cv2.line(frame, (0, 124), (width, 124), (base - 15,) * 3, 2)

    phase = 2 * math.pi * t / 40
    cx = 65 + (variant % 3) * 5
    gait = 0.0
    if action == "walking":
        cx += int(1.25 * t); gait = math.sin(phase * 1.5)
    elif action == "jogging":
        cx += int(1.85 * t); gait = math.sin(phase * 2.5)
    elif action == "running":
        cx += int(2.55 * t); gait = math.sin(phase * 3.4)
    cy = 78 + (int(abs(gait) * 2) if action in {"jogging", "running"} else 0)
    scale = 0.86 + .07 * (variant % 3)
    ink_value = 25 if scenario != 3 else 225
    ink = (ink_value,) * 3
    head = (cx, int(cy - 31 * scale))
    shoulder = int(cy - 20 * scale)
    hip = int(cy + 8 * scale)
    cv2.circle(frame, head, max(4, int(6 * scale)), ink, -1, cv2.LINE_AA)
    cv2.line(frame, (cx, shoulder), (cx, hip), ink, max(3, int(6 * scale)), cv2.LINE_AA)

    if action in {"walking", "jogging", "running"}:
        stride = (13 if action == "walking" else 17 if action == "jogging" else 22) * gait
        cv2.line(frame, (cx, hip), (int(cx + stride), int(cy + 34 * scale)), ink, 4, cv2.LINE_AA)
        cv2.line(frame, (cx, hip), (int(cx - stride), int(cy + 34 * scale)), ink, 4, cv2.LINE_AA)
        arm = -stride * .68
        cv2.line(frame, (cx, shoulder), (int(cx + arm), cy), ink, 4, cv2.LINE_AA)
        cv2.line(frame, (cx, shoulder), (int(cx - arm), cy), ink, 4, cv2.LINE_AA)
    elif action == "boxing":
        punch = (math.sin(phase * 3) + 1) / 2
        cv2.line(frame, (cx, shoulder), (int(cx + 10 + 20 * punch), shoulder - 2), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, shoulder + 4), (int(cx + 22 - 12 * punch), shoulder + 8), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, hip), (cx - 6, cy + 35), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, hip), (cx + 8, cy + 35), ink, 5, cv2.LINE_AA)
    elif action == "handclapping":
        gap = 4 + int(18 * abs(math.sin(phase * 2.4)))
        cv2.line(frame, (cx - 2, shoulder), (cx - gap, shoulder + 8), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx + 2, shoulder), (cx + gap, shoulder + 8), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, hip), (cx - 7, cy + 35), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, hip), (cx + 7, cy + 35), ink, 5, cv2.LINE_AA)
    else:  # handwaving
        angle = math.sin(phase * 2.2)
        hand_x = int(cx + 18 * angle)
        hand_y = int(shoulder - 20 - 7 * math.cos(phase * 2.2))
        cv2.line(frame, (cx, shoulder), (hand_x, hand_y), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, shoulder), (cx - 16, shoulder + 15), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, hip), (cx - 7, cy + 35), ink, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, hip), (cx + 7, cy + 35), ink, 5, cv2.LINE_AA)
    return frame


def generate_demo_dataset(root: str | Path, per_class: int = 8) -> Path:
    root = Path(root)
    for class_id, action in CLASS_NAMES.items():
        for sample in range(per_class):
            folder = root / action / f"{action}_{sample + 1:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(10_000 * class_id + sample)
            frames = [_person_frame(action, index, rng, sample) for index in range(40)]
            boxes = []
            for index, frame in enumerate(frames):
                cv2.imwrite(str(folder / f"frame_{index + 1:03d}.png"), frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                background = int(np.median(gray[:15]))
                mask = np.uint8(np.abs(gray.astype(int) - background) > 35) * 255
                points = cv2.findNonZero(mask)
                boxes.append(cv2.boundingRect(points) if points is not None else (0, 0, 1, 1))
            x, y, w, h = boxes[19]
            height, width = frames[19].shape[:2]
            annotation = f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}\n"
            (folder / "annotation.txt").write_text(annotation, encoding="utf-8")
    return root

