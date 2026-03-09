from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class SklearnBaseline:
    def __init__(self, name: str, model_kind: str):
        self.name = name
        if model_kind == "logreg":
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        elif model_kind == "knn":
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(model_kind)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        pred = self.model.predict(x)
        return pred.astype(float)

    def update(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        x_comb = np.concatenate([x_new], axis=0)
        y_comb = np.concatenate([y_new], axis=0)
        self.model.fit(x_comb, y_comb)
