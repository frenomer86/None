# DART+AGIL for Encrypted Traffic Drift Detection

This repository provides an executable, research-oriented pipeline for encrypted traffic classification under temporal drift.

## Scientific honesty constraints
- No fabricated metrics, tables, or claims.
- No synthetic/mock datasets in experiment scripts.
- Every generated artifact is from real execution on user-provided real datasets.
- If not executed: `Result pending real execution.`

## Implemented scope
- Proposed method: DART+AGIL with online transfer update.
- Baseline suite (implementation proxies for manuscript comparison):
  - `cdda_md_proxy`
  - `m3s_upd_proxy`
  - `cbr_proxy`
  - `ssmd_proxy`
  - `if_dr`
- Full benchmark script for:
  - standard metrics,
  - temporal drift windows,
  - obfuscation robustness (IDP/IBP/APR/INP proxies),
  - few-shot analysis,
  - table export (`.csv`, `.tex`),
  - figure export (`.pdf`, black/white pattern-oriented style).
- Dedicated ablation runner for `full`, `no_dart`, `no_agil`, `no_online_tl`.

## Dataset handling
- The preprocessing stage supports automatic dataset resolution via `dataset.auto_download`:
  - NSL-KDD: fully auto-download + extraction supported.
  - CICIDS2017/CICDDoS2019: official pages are not direct CSV archives; script raises an actionable error and requires local CSV path in config.

Set dataset paths and options in `configs/*.yaml`.

## End-to-end commands
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Preprocess (auto-download where supported)
python scripts/preprocess_dataset.py --config configs/nslkdd.yaml

# Proposed method train/eval
python scripts/train.py --config configs/cicids2017.yaml --experiment full
python scripts/evaluate.py --config configs/cicids2017.yaml --checkpoint outputs/checkpoints/cicids2017_full_full_best.pt --experiment full

# Full method+baseline benchmark, all tables and figures
python scripts/run_benchmark.py --config configs/cicids2017.yaml

# Full ablation suite + table/figure
python scripts/run_ablations.py --config configs/cicids2017.yaml
```

## Main outputs
- `outputs/results/tables/*.csv`
- `outputs/results/tables/*.tex`
- `outputs/figures/*.pdf`
- `outputs/results/*benchmark_manifest.json`
- `outputs/results/*ablation_manifest.json`
