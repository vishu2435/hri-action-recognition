from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .constants import CLASS_NAMES
from .features import FEATURE_NAMES, extract_features
from .vision import TrackResult


@dataclass(frozen=True)
class Prediction:
    class_id: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    track: TrackResult


class ActionRecognizer:
    def __init__(self, estimator: RandomForestClassifier | None = None, metadata: dict[str, Any] | None = None):
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=350, max_depth=12, min_samples_leaf=1,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.metadata = metadata or {}

    def fit(self, frame_sequences: list[list[np.ndarray]], labels: list[int]) -> "ActionRecognizer":
        features = np.vstack([extract_features(frames)[0] for frames in frame_sequences])
        self.estimator.fit(features, labels)
        return self

    def fit_features(self, features: np.ndarray, labels: np.ndarray) -> "ActionRecognizer":
        self.estimator.fit(features, labels)
        return self

    def predict(self, frames: list[np.ndarray]) -> Prediction:
        features, track = extract_features(frames)
        probabilities = self.estimator.predict_proba(features.reshape(1, -1))[0]
        classes = self.estimator.classes_.astype(int)
        winner = int(np.argmax(probabilities))
        class_id = int(classes[winner])
        by_name = {CLASS_NAMES[int(cid)]: float(prob) for cid, prob in zip(classes, probabilities)}
        return Prediction(class_id, CLASS_NAMES[class_id], float(probabilities[winner]), by_name, track)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "features": FEATURE_NAMES,
                     "class_names": CLASS_NAMES, "metadata": self.metadata, "version": 2}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ActionRecognizer":
        payload = joblib.load(path)
        if payload.get("features") != FEATURE_NAMES:
            raise ValueError("Model feature schema is incompatible with this version")
        return cls(payload["estimator"], payload.get("metadata", {}))
