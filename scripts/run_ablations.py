#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

from driftdart.utils.config import load_config, ensure_dirs
from driftdart.reporting.tables import save_csv_tables, save_latex_table
from driftdart.visualization.plots import save_pattern_bar_pdf


def _run(cmd: list[str]):
    proc = subprocess.run(cmd, check=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_dirs(cfg['paths']['output_root'])

    experiments = cfg.get('ablations', ['full', 'no_dart', 'no_agil', 'no_online_tl'])

    for exp in experiments:
        _run([sys.executable, 'scripts/train.py', '--config', args.config, '--experiment', exp])
        ckpt = str(Path(dirs['checkpoints']) / f"{cfg['experiment_name']}_{exp}_best.pt")
        _run([sys.executable, 'scripts/evaluate.py', '--config', args.config, '--checkpoint', ckpt, '--experiment', exp])

    rows = []
    for exp in experiments:
        metrics_path = Path(dirs['results']) / f"{cfg['experiment_name']}_{exp}_metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding='utf-8'))
        t = m.get('test', {})
        rows.append({
            'ablation': exp,
            'accuracy': t.get('accuracy'),
            'precision': t.get('precision'),
            'recall': t.get('recall'),
            'f1': t.get('f1'),
            'auc': t.get('auc'),
        })

    df = pd.DataFrame(rows)
    tables_dir = Path(dirs['results']) / 'tables'
    save_csv_tables(df, tables_dir, f"{cfg['experiment_name']}_ablation")
    save_latex_table(df, tables_dir, f"{cfg['experiment_name']}_ablation", 'Ablation results', 'tab:ablation')

    figs_dir = Path(dirs['figures'])
    if len(df):
        save_pattern_bar_pdf(df['ablation'].tolist(), df['f1'].fillna(0).tolist(), figs_dir / 'ablation_f1_bar.pdf', 'F1-score', 'Ablation Comparison')

    manifest = {
        'experiment': cfg['experiment_name'],
        'ablations': experiments,
        'ablation_table': str(tables_dir / f"{cfg['experiment_name']}_ablation.csv"),
        'ablation_figure': str(figs_dir / 'ablation_f1_bar.pdf'),
        'result_status': 'Result pending real execution.'
    }
    out = Path(dirs['results']) / f"{cfg['experiment_name']}_ablation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Saved ablation outputs: {out}")


if __name__ == '__main__':
    main()
