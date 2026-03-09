#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np

from driftdart.utils.config import ensure_dirs, load_config
from driftdart.data.preprocessing import load_and_prepare_dataframe, temporal_split, fit_transform_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_dirs(cfg['paths']['output_root'])

    dcfg = cfg['dataset']
    pcfg = cfg['preprocessing']

    df = load_and_prepare_dataframe(
        cfg['paths']['raw_csv'],
        dcfg['label_col'],
        dcfg.get('timestamp_col'),
        dcfg['positive_regex'],
        dcfg.get('drop_columns', []),
        dcfg.get('max_rows')
    )

    tr, va, te = temporal_split(df, pcfg['train_ratio'], pcfg['val_ratio'])
    xtr, ytr, xva, yva, xte, yte, artifacts = fit_transform_features(tr, va, te, clip_iqr=pcfg.get('clip_iqr', True))

    np.savez_compressed(dirs['processed'] / f"{cfg['experiment_name']}.npz", x_train=xtr, y_train=ytr, x_val=xva, y_val=yva, x_test=xte, y_test=yte)
    joblib.dump(artifacts, dirs['processed'] / f"{cfg['experiment_name']}_artifacts.joblib")
    print(f"Saved processed dataset: {dirs['processed'] / (cfg['experiment_name'] + '.npz')}")


if __name__ == '__main__':
    main()
