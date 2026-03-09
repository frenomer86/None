from __future__ import annotations

import numpy as np


def _insert_dummy_like_noise(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < p
    noise = rng.normal(0.0, 0.03, size=x.shape)
    out = x.copy()
    out[mask] = np.clip(out[mask] + noise[mask], 0.0, 1.0)
    return out


def _insert_benign_mix(x: np.ndarray, benign_pool: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    if len(benign_pool) == 0:
        return out
    idx = rng.integers(0, len(benign_pool), size=len(x))
    mix = benign_pool[idx]
    alpha = (rng.random((len(x), 1)) < p).astype(float) * 0.5
    out = np.clip((1 - alpha) * out + alpha * mix, 0.0, 1.0)
    return out


def _alter_packet_rate_proxy(x: np.ndarray, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    scale = rng.uniform(1.0 - magnitude, 1.0 + magnitude, size=(len(x), 1))
    return np.clip(x * scale, 0.0, 1.0)


def _insert_packet_noise_proxy(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < p
    noise = rng.uniform(-0.1, 0.1, size=x.shape)
    out = x.copy()
    out[mask] = np.clip(out[mask] + noise[mask], 0.0, 1.0)
    return out


def apply_obfuscation(x: np.ndarray, y: np.ndarray, strategy: str, level: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_mal = x[y == 1]
    x_ben = x[y == 0]
    x_out = x.copy()

    if strategy == "idp":
        x_out[y == 1] = _insert_dummy_like_noise(x_mal, level, rng)
    elif strategy == "ibp":
        x_out[y == 1] = _insert_benign_mix(x_mal, x_ben, level, rng)
    elif strategy == "apr":
        x_out[y == 1] = _alter_packet_rate_proxy(x_mal, level, rng)
    elif strategy == "inp":
        x_out[y == 1] = _insert_packet_noise_proxy(x_mal, level, rng)
    else:
        raise ValueError(f"Unknown obfuscation strategy: {strategy}")
    return x_out
