from __future__ import annotations

import numpy as np


def temporal_windows(x: np.ndarray, y: np.ndarray, n_windows: int = 5):
    n = len(x)
    w = max(1, n // n_windows)
    out = []
    for i in range(n_windows):
        s = i * w
        e = (i + 1) * w if i < n_windows - 1 else n
        out.append((x[s:e], y[s:e], i))
    return out


def drift_degradation(initial_metric: float, drifted_metric: float) -> float:
    return float(initial_metric - drifted_metric)
