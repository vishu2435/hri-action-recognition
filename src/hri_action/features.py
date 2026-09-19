# Author: Himashi
from __future__ import annotations

import cv2
import numpy as np

from .vision import TrackResult, track_person


FEATURE_NAMES = [
    "centroid_speed_mean", "centroid_speed_std", "centroid_speed_max", "net_displacement",
    "bbox_width_mean", "bbox_height_mean", "bbox_area_mean", "bbox_area_std",
    "motion_area_mean", "motion_area_std", "motion_area_frequency", "flow_magnitude_mean",
    "flow_magnitude_std", "flow_magnitude_p90", "horizontal_flow", "vertical_flow",
    "upper_motion", "lower_motion", "upper_lower_ratio", "upper_frequency",
    "lower_frequency", "direction_0", "direction_1", "direction_2", "direction_3",
    "direction_4", "direction_5", "direction_6", "direction_7", "inward_flow",
    "outward_flow", "aspect_ratio",
]


def _dominant_frequency(values: np.ndarray) -> float:
    centered = values - values.mean()
    spectrum = np.abs(np.fft.rfft(centered))
    if len(spectrum) <= 1 or float(spectrum[1:].sum()) < 1e-8:
        return 0.0
    index = int(np.argmax(spectrum[1:]) + 1)
    return index / max(1, len(values) - 1)


def extract_features(frames: list[np.ndarray], track: TrackResult | None = None) -> tuple[np.ndarray, TrackResult]:
    track = track or track_person(frames)
    h, w = frames[0].shape[:2]
    boxes = np.asarray(track.boxes, dtype=np.float32)
    centers = boxes[:, :2] + boxes[:, 2:] / 2
    velocities = np.linalg.norm(np.diff(centers, axis=0), axis=1) / max(h, w)
    net = np.linalg.norm(centers[-1] - centers[0]) / max(h, w)
    widths, heights = boxes[:, 2] / w, boxes[:, 3] / h
    areas = widths * heights
    motion_areas = np.asarray([np.count_nonzero(mask) / (h * w) for mask in track.motion_masks])

    flow_magnitudes: list[float] = []
    angles: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    upper: list[float] = []
    lower: list[float] = []
    inward_values: list[float] = []
    outward_values: list[float] = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for index, frame in enumerate(frames[1:], start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        x, y, bw, bh = track.boxes[index]
        roi_mag = magnitude[y:y + bh, x:x + bw]
        roi_ang = angle[y:y + bh, x:x + bw]
        if roi_mag.size:
            cutoff = np.percentile(roi_mag, 60)
            active = roi_mag >= max(.1, cutoff)
            selected_mag = roi_mag[active]
            selected_ang = roi_ang[active]
            flow_magnitudes.extend(selected_mag.tolist())
            angles.append(selected_ang)
            magnitudes.append(selected_mag)
            split = max(1, bh // 2)
            upper.append(float(roi_mag[:split].mean()))
            lower.append(float(roi_mag[split:].mean()) if split < bh else 0.0)
            yy, xx = np.mgrid[y:y + bh, x:x + bw]
            cx = x + bw / 2
            radial_x = xx - cx
            radial_dot = flow[y:y + bh, x:x + bw, 0] * radial_x
            inward_values.append(float(np.mean(np.maximum(-radial_dot, 0))))
            outward_values.append(float(np.mean(np.maximum(radial_dot, 0))))
        prev = gray

    mags = np.asarray(flow_magnitudes or [0.0], dtype=np.float32)
    if angles:
        all_angles = np.concatenate(angles)
        all_mags = np.concatenate(magnitudes)
        histogram, _ = np.histogram(all_angles, bins=8, range=(0, 2 * np.pi), weights=all_mags)
        histogram = histogram / max(float(histogram.sum()), 1e-8)
    else:
        histogram = np.zeros(8)
    upper_array = np.asarray(upper or [0.0])
    lower_array = np.asarray(lower or [0.0])
    horizontal = float(histogram[[0, 3, 4, 7]].sum())
    vertical = float(histogram[[1, 2, 5, 6]].sum())
    vector = np.asarray([
        velocities.mean(), velocities.std(), velocities.max(initial=0), net,
        widths.mean(), heights.mean(), areas.mean(), areas.std(),
        motion_areas.mean(), motion_areas.std(), _dominant_frequency(motion_areas),
        mags.mean(), mags.std(), np.percentile(mags, 90), horizontal, vertical,
        upper_array.mean(), lower_array.mean(), upper_array.mean() / max(lower_array.mean(), 1e-6),
        _dominant_frequency(upper_array), _dominant_frequency(lower_array), *histogram.tolist(),
        np.mean(inward_values or [0.0]), np.mean(outward_values or [0.0]),
        np.mean(widths / np.maximum(heights, 1e-6)),
    ], dtype=np.float32)
    return np.nan_to_num(vector), track

