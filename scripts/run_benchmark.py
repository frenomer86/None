#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from driftdart.utils.config import load_config, ensure_dirs
from driftdart.utils.seed import set_seed
from driftdart.baselines.registry import available_baselines, build_baseline
from driftdart.baselines.dart_wrapper import DARTAGILWrapper
from driftdart.eval.metrics import classification_metrics
from driftdart.eval.obfuscation import apply_obfuscation
from driftdart.eval.drift import temporal_windows
from driftdart.eval.fewshot import run_few_shot
from driftdart.reporting.tables import save_csv_tables, save_latex_table
from driftdart.visualization.plots import save_pattern_bar_pdf, save_line_pdf, save_heatmap_pdf


def _evaluate_method(name, model, x_train, y_train, x_val, y_val, x_test, y_test, cfg):
    t0 = time.perf_counter()
    if name == 'dart_agil':
        model.fit(x_train, y_train, x_val, y_val)
    else:
        model.fit(x_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob = model.predict_proba(x_test)
    infer_s = time.perf_counter() - t1
    base = classification_metrics(y_test, prob, cfg['evaluation']['threshold'])
    base['train_seconds'] = train_s
    base['inference_ms_per_flow'] = (infer_s / max(1, len(x_test))) * 1000.0

    windows = temporal_windows(x_test, y_test, n_windows=5)
    drift_rows = []
    for xw, yw, widx in windows:
        if len(yw) == 0:
            continue
        pw = model.predict_proba(xw)
        m = classification_metrics(yw, pw, cfg['evaluation']['threshold'])
        drift_rows.append({'method': name, 'window': widx, 'f1': m['f1'], 'auc': m['auc']})

    obf_rows = []
    obf_specs = [('idp', [0.1, 0.2, 0.3]), ('ibp', [0.1, 0.3, 0.5]), ('apr', [0.5]), ('inp', [0.5])]
    for strat, levels in obf_specs:
        for lv in levels:
            x_obf = apply_obfuscation(x_test, y_test, strat, lv, seed=cfg['seed'])
            p_obf = model.predict_proba(x_obf)
            mm = classification_metrics(y_test, p_obf, cfg['evaluation']['threshold'])
            obf_rows.append({'method': name, 'strategy': strat, 'level': lv, 'f1': mm['f1'], 'auc': mm['auc']})

    few_rows = []
    for n in cfg['evaluation']['few_shot_ns']:
        if name == 'dart_agil':
            fs = run_few_shot(model.model, x_train, y_train, x_test, y_test, n, cfg['evaluation']['few_shot_repeats'], model.device)
        else:
            fs = {'mean_f1': None, 'std_f1': None, 'n': n, 'repeats': 0}
        few_rows.append({'method': name, 'n': n, 'mean_f1': fs['mean_f1'], 'std_f1': fs['std_f1'], 'repeats': fs['repeats']})

    return base, drift_rows, obf_rows, few_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--methods', nargs='*', default=available_baselines())
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    dirs = ensure_dirs(cfg['paths']['output_root'])
    d = np.load(dirs['processed'] / f"{cfg['experiment_name']}.npz")

    x_train, y_train = d['x_train'], d['y_train']
    x_val, y_val = d['x_val'], d['y_val']
    x_test, y_test = d['x_test'], d['y_test']

    summary, drift_all, obf_all, few_all = [], [], [], []

    for name in args.methods:
        if name == 'dart_agil':
            model = DARTAGILWrapper(cfg, name=name)
        else:
            model = build_baseline(name)

        base, drift_rows, obf_rows, few_rows = _evaluate_method(name, model, x_train, y_train, x_val, y_val, x_test, y_test, cfg)
        base['method'] = name
        summary.append(base)
        drift_all.extend(drift_rows)
        obf_all.extend(obf_rows)
        few_all.extend(few_rows)

    summary_df = pd.DataFrame(summary)[['method', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'train_seconds', 'inference_ms_per_flow']]
    drift_df = pd.DataFrame(drift_all)
    obf_df = pd.DataFrame(obf_all)
    few_df = pd.DataFrame(few_all)

    tables_dir = Path(dirs['results']) / 'tables'
    figs_dir = Path(dirs['figures'])

    save_csv_tables(summary_df, tables_dir, f"{cfg['experiment_name']}_summary")
    save_csv_tables(drift_df, tables_dir, f"{cfg['experiment_name']}_drift")
    save_csv_tables(obf_df, tables_dir, f"{cfg['experiment_name']}_obfuscation")
    save_csv_tables(few_df, tables_dir, f"{cfg['experiment_name']}_fewshot")

    save_latex_table(summary_df, tables_dir, f"{cfg['experiment_name']}_summary", 'Overall test performance by method', 'tab:summary')
    save_latex_table(obf_df, tables_dir, f"{cfg['experiment_name']}_obfuscation", 'Obfuscation robustness by method', 'tab:obf')
    save_latex_table(few_df, tables_dir, f"{cfg['experiment_name']}_fewshot", 'Few-shot results (where applicable)', 'tab:few')

    save_pattern_bar_pdf(summary_df['method'].tolist(), summary_df['f1'].tolist(), figs_dir / 'benchmark_f1_bar.pdf', 'F1-score', 'Method Comparison (F1)')

    drift_plot_df = drift_df.pivot_table(index='window', columns='method', values='f1', aggfunc='mean').sort_index()
    save_line_pdf(drift_plot_df.index.tolist(), {c: drift_plot_df[c].fillna(0).tolist() for c in drift_plot_df.columns}, figs_dir / 'drift_over_time.pdf', 'F1-score', 'Temporal Drift Performance')

    obf_plot = obf_df.copy()
    obf_plot['cond'] = obf_plot['strategy'] + '@' + obf_plot['level'].astype(str)
    heat = obf_plot.pivot_table(index='method', columns='cond', values='f1', aggfunc='mean').fillna(0)
    save_heatmap_pdf(heat.values, heat.index.tolist(), heat.columns.tolist(), figs_dir / 'obfuscation_heatmap.pdf', 'Obfuscation Robustness (F1)')

    out_json = Path(dirs['results']) / f"{cfg['experiment_name']}_benchmark_manifest.json"
    out_json.write_text(json.dumps({
        'experiment': cfg['experiment_name'],
        'methods': args.methods,
        'tables_dir': str(tables_dir),
        'figures_dir': str(figs_dir),
        'result_status': 'Result pending real execution.'
    }, indent=2), encoding='utf-8')
    print(f"Saved benchmark outputs: {out_json}")


if __name__ == '__main__':
    main()
