from __future__ import annotations

from .sklearn_models import SklearnBaseline
from .torch_mlp import TorchMLPBaseline
from .ifdr import IFDRBaseline


def available_baselines() -> list[str]:
    return [
        "cdda_md_proxy",
        "m3s_upd_proxy",
        "cbr_proxy",
        "ssmd_proxy",
        "if_dr",
        "dart_agil",
    ]


def build_baseline(name: str):
    if name == "cdda_md_proxy":
        return TorchMLPBaseline(name=name, hidden=(256, 128), dropout=0.2)
    if name == "m3s_upd_proxy":
        return TorchMLPBaseline(name=name, hidden=(512, 256), dropout=0.3)
    if name == "cbr_proxy":
        return SklearnBaseline(name=name, model_kind="knn")
    if name == "ssmd_proxy":
        return SklearnBaseline(name=name, model_kind="logreg")
    if name == "if_dr":
        return IFDRBaseline(name=name)
    raise ValueError(f"Unknown baseline: {name}")
