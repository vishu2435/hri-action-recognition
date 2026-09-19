# Author: Samindu
from __future__ import annotations

import numpy as np

from hri_action.demo import generate_demo_dataset
from hri_action.dataset import Annotation, SequenceRecord, discover_sequences, load_frames, normalized_bbox_to_pixels
from hri_action.experiment import stratified_split
from hri_action.features import FEATURE_NAMES, extract_features
from hri_action.validation import validate_dataset
from hri_action.vision import bbox_iou


def test_demo_dataset_and_features(tmp_path):
    generate_demo_dataset(tmp_path, per_class=1)
    records = discover_sequences(tmp_path)
    assert len(records) == 6
    frames = load_frames(records[0])
    features, track = extract_features(frames)
    assert len(features) == len(FEATURE_NAMES) == 32
    assert np.isfinite(features).all()
    assert len(track.boxes) == 40


def test_bbox_helpers():
    assert normalized_bbox_to_pixels((.5, .5, .5, .5), 100, 80) == (25, 20, 50, 40)
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 4, 4)) == 0.0


def test_proposal_dataset_contract_and_split(tmp_path):
    records = []
    for class_id in range(1, 7):
        for scenario in range(1, 5):
            for sample in range(3):
                path = tmp_path / f"class_{class_id}" / f"d{scenario}" / f"sequence_{sample}"
                records.append(SequenceRecord(
                    path, tuple(path / f"frame_{i:03d}.png" for i in range(40)),
                    Annotation(class_id, (.5, .5, .2, .6)), 40,
                ))
    report = validate_dataset(records, strict_proposal=True)
    assert report.valid
    train, test = stratified_split(records, seed=42)
    assert len(train) == 48
    assert len(test) == 24
    assert {record.scenario for record in test} == {"d1", "d2", "d3", "d4"}
