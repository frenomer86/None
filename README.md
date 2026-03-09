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

## Dataset requirement
Use real, local dataset exports only:
- CICIDS2017 CSV
- CICDDoS2019 CSV
- NSL-KDD CSV/TXT converted to table

Set `paths.raw_csv` in `configs/*.yaml`.

## End-to-end commands
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/preprocess_dataset.py --config configs/cicids2017.yaml
python scripts/train.py --config configs/cicids2017.yaml --experiment full
python scripts/evaluate.py --config configs/cicids2017.yaml --checkpoint outputs/checkpoints/cicids2017_full_full_best.pt --experiment full

# Full method+baseline benchmark, all tables and figures
python scripts/run_benchmark.py --config configs/cicids2017.yaml
```

## Main outputs
- `outputs/results/tables/*.csv`
- `outputs/results/tables/*.tex`
- `outputs/figures/*.pdf`
- `outputs/results/*benchmark_manifest.json`
