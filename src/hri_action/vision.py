from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Box = tuple[int, int, int, int]


@dataclass
class TrackResult:
    boxes: list[Box]
    masks: list[np.ndarray]
    motion_masks: list[np.ndarray]


def _largest_component(mask: np.ndarray, min_area: int) -> tuple[Box | None, np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None, np.zeros_like(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        return None, np.zeros_like(mask)
    selected = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[selected]
    return (int(x), int(y), int(w), int(h)), np.uint8(labels == selected) * 255


def _clip_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))


def track_person(frames: list[np.ndarray]) -> TrackResult:
    """Track the single actor using temporal foreground energy and connected components."""
    if len(frames) < 3:
        raise ValueError("At least three frames are required")
    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames]
    h, w = grays[0].shape
    min_area = max(12, int(h * w * 0.0007))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    diffs: list[np.ndarray] = []
    appearances: list[np.ndarray] = []
    for index, gray in enumerate(grays):
        before = grays[max(0, index - 1)]
        after = grays[min(len(grays) - 1, index + 1)]
        diff = cv2.max(cv2.absdiff(gray, before), cv2.absdiff(gray, after))
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7, iterations=2)
        diffs.append(mask)

        border = np.concatenate((gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()))
        background_level = float(np.median(border))
        contrast = np.uint8(np.abs(gray.astype(np.float32) - background_level))
        threshold = max(12.0, float(np.percentile(contrast, 91)) * .55)
        appearance = np.uint8(contrast >= threshold) * 255
        appearance = cv2.morphologyEx(appearance, cv2.MORPH_OPEN, kernel3)
        appearance = cv2.morphologyEx(appearance, cv2.MORPH_CLOSE, kernel7, iterations=2)
        appearances.append(appearance)

    # A sequence-level motion envelope stabilizes stationary-in-place actions.
    envelope = np.max(np.stack(diffs), axis=0)
    envelope = cv2.dilate(envelope, kernel7, iterations=2)
    env_box, env_mask = _largest_component(envelope, min_area)
    if env_box is None:
        env_box = (int(w * .3), int(h * .1), int(w * .4), int(h * .8))
        env_mask = np.zeros((h, w), np.uint8)
        x, y, bw, bh = env_box
        env_mask[y:y + bh, x:x + bw] = 255

    boxes: list[Box] = []
    masks: list[np.ndarray] = []
    previous = np.array(env_box, dtype=float)
    ex, ey, ew, eh = env_box
    gate = np.zeros((h, w), np.uint8)
    margin_x, margin_y = max(10, ew), max(14, eh)
    gx1, gy1 = max(0, ex - margin_x), max(0, ey - margin_y)
    gx2, gy2 = min(w, ex + ew + margin_x), min(h, ey + eh + margin_y)
    gate[gy1:gy2, gx1:gx2] = 255
    for diff, appearance in zip(diffs, appearances):
        combined = cv2.bitwise_or(appearance, cv2.dilate(diff, kernel7, iterations=1))
        local = cv2.bitwise_and(combined, gate)
        box, component = _largest_component(local, min_area)
        if box is None:
            proposal = previous
            component = local
        else:
            proposal = np.array(box, dtype=float)
            # Expand sparse motion to a person-like region, then smooth the track.
            proposal[0] -= proposal[2] * .15
            proposal[1] -= proposal[3] * .12
            proposal[2] *= 1.30
            proposal[3] *= 1.24
            proposal = .35 * proposal + .65 * previous
        previous = proposal
        clipped = _clip_box(tuple(int(round(v)) for v in proposal), w, h)
        boxes.append(clipped)
        silhouette = np.zeros((h, w), np.uint8)
        x, y, bw, bh = clipped
        silhouette[y:y + bh, x:x + bw] = component[y:y + bh, x:x + bw]
        masks.append(silhouette)
    return TrackResult(boxes, masks, diffs)


def bbox_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def annotate_frame(frame: np.ndarray, box: Box, label: str, confidence: float | None = None) -> np.ndarray:
    output = frame.copy()
    x, y, w, h = box
    color = (57, 222, 166)
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    text = label if confidence is None else f"{label}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
    top = max(0, y - th - 12)
    cv2.rectangle(output, (x, top), (min(output.shape[1], x + tw + 12), y), color, -1)
    cv2.putText(output, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, .55, (10, 18, 27), 2, cv2.LINE_AA)
    return output
