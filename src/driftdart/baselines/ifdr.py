from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier


class IFDRBaseline:
    """Isolation-Forest-triggered selective retraining with Random Forest classifier."""

    def __init__(self, name: str):
        self.name = name
        self.detector = IsolationForest(contamination=0.05, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        self._x_ref = None
        self._y_ref = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self._x_ref = x_train.copy()
        self._y_ref = y_train.copy()
        self.detector.fit(x_train)
        self.classifier.fit(x_train, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(x)[:, 1]

    def update(self, x_new: np.ndarray, y_new: np.ndarray, drift_threshold: float = 0.2) -> bool:
        flags = self.detector.predict(x_new)
        drift_ratio = float(np.mean(flags == -1))
        if drift_ratio >= drift_threshold:
            x_fit = np.concatenate([self._x_ref, x_new], axis=0)
            y_fit = np.concatenate([self._y_ref, y_new], axis=0)
            self.classifier.fit(x_fit, y_fit)
            self._x_ref, self._y_ref = x_fit, y_fit
            self.detector.fit(x_fit)
            return True
        return False
