#!/usr/bin/env python3
from __future__ import annotations

import argparse
from driftdart.visualization.plots import plot_from_metrics_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    plot_from_metrics_json(args.results, args.outdir)
    print(f"Saved PDF figures to: {args.outdir}")


if __name__ == '__main__':
    main()
