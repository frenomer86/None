from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreprocessArtifacts:
    scaler: MinMaxScaler
    feature_columns: list[str]


def _to_binary_labels(series: pd.Series, positive_regex: str) -> pd.Series:
    return series.astype(str).str.contains(positive_regex, regex=True).astype(int)


def _iqr_clip(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        q1 = out[c].quantile(0.25)
        q3 = out[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def load_and_prepare_dataframe(
    csv_path: str,
    label_col: str,
    timestamp_col: Optional[str],
    positive_regex: str,
    drop_columns: list[str],
    max_rows: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False, nrows=max_rows)
    for c in drop_columns:
        if c in df.columns:
            df = df.drop(columns=c)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in dataset columns")

    df["target"] = _to_binary_labels(df[label_col], positive_regex)

    if timestamp_col and timestamp_col in df.columns:
        df["_time"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values("_time")
    else:
        df = df.reset_index().rename(columns={"index": "_time_idx"})
        df = df.sort_values("_time_idx")

    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "target" not in numeric.columns:
        numeric["target"] = df["target"].values
    return numeric


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    tr_end = int(n * train_ratio)
    va_end = tr_end + int(n * val_ratio)
    return df.iloc[:tr_end], df.iloc[tr_end:va_end], df.iloc[va_end:]


def fit_transform_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clip_iqr: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessArtifacts]:
    feature_cols = [c for c in train_df.columns if c != "target"]

    tr = train_df.copy()
    va = val_df.copy()
    te = test_df.copy()

    if clip_iqr:
        tr = _iqr_clip(tr, feature_cols)
        va = _iqr_clip(va, feature_cols)
        te = _iqr_clip(te, feature_cols)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(tr[feature_cols].values)
    x_val = scaler.transform(va[feature_cols].values)
    x_test = scaler.transform(te[feature_cols].values)

    y_train = tr["target"].to_numpy(dtype=np.int64)
    y_val = va["target"].to_numpy(dtype=np.int64)
    y_test = te["target"].to_numpy(dtype=np.int64)

    return x_train, y_train, x_val, y_val, x_test, y_test, PreprocessArtifacts(scaler=scaler, feature_columns=feature_cols)
