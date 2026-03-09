from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(output_root: str | Path) -> Dict[str, Path]:
    base = Path(output_root)
    dirs = {
        "base": base,
        "processed": base / "processed",
        "checkpoints": base / "checkpoints",
        "results": base / "results",
        "figures": base / "figures",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
