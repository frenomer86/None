from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_csv_tables(df: pd.DataFrame, outdir: str | Path, stem: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def save_latex_table(df: pd.DataFrame, outdir: str | Path, stem: str, caption: str, label: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.tex"
    tex = df.to_latex(index=False, escape=False, caption=caption, label=label)
    path.write_text(tex, encoding="utf-8")
    return path
