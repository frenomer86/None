from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np


def _style_bw():
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'


def save_pattern_bar_pdf(labels, values, outpath: str | Path, ylabel: str, title: str):
    _style_bw()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    hatches = ['/', '\\\\', 'x', '-', '+', 'o', '.', '*']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(labels)), values, color='white', edgecolor='black', linewidth=1.2)
    for i, b in enumerate(bars):
        b.set_hatch(hatches[i % len(hatches)])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, format='pdf')
    plt.close(fig)


def save_line_pdf(x, y_series: dict[str, list[float]], outpath: str | Path, ylabel: str, title: str):
    _style_bw()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    markers = ['o', 's', '^', 'd', 'x', '*']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, y) in enumerate(y_series.items()):
        ax.plot(x, y, marker=markers[i % len(markers)], color='black', linewidth=1.2, markersize=4, label=name, linestyle=['-', '--', '-.', ':'][i % 4])

    ax.set_xlabel('Window')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, format='pdf')
    plt.close(fig)


def save_heatmap_pdf(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], outpath: str | Path, title: str):
    _style_bw()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(matrix, cmap='Greys', aspect='auto')
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center', color='black', fontsize=7)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, format='pdf')
    plt.close(fig)


def plot_from_metrics_json(metrics_json: str | Path, outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(metrics_json, 'r', encoding='utf-8') as f:
        m = json.load(f)

    base = m.get('test', {})
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    vals = [base.get(k, 0.0) if base.get(k) is not None else 0.0 for k in keys]
    save_pattern_bar_pdf(keys, vals, outdir / 'test_metrics_bar.pdf', 'Score', 'Test Metrics')

    fs = m.get('few_shot', [])
    if fs:
        labels = [f"N={x['n']}" for x in fs if x.get('mean_f1') is not None]
        values = [x['mean_f1'] for x in fs if x.get('mean_f1') is not None]
        if labels:
            save_pattern_bar_pdf(labels, values, outdir / 'fewshot_bar.pdf', 'F1-score', 'Few-shot Performance')
