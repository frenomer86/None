#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from driftdart.utils.config import ensure_dirs, load_config
from driftdart.utils.seed import set_seed
from driftdart.utils.logging import build_logger
from driftdart.data.dataset import FlowDataset
from driftdart.models.dart_agil import DARTAGIL
from driftdart.training.engine import train, evaluate_model
from driftdart.training.online_transfer import online_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--experiment', default='full', choices=['full', 'no_dart', 'no_agil', 'no_online_tl'])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg['ablation'] = args.experiment
    set_seed(cfg['seed'])
    dirs = ensure_dirs(cfg['paths']['output_root'])
    logger = build_logger(dirs['logs'] / f"train_{cfg['experiment_name']}_{args.experiment}.log")

    data_path = dirs['processed'] / f"{cfg['experiment_name']}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run preprocess_dataset.py first.")

    d = np.load(data_path)
    x_train, y_train = d['x_train'], d['y_train']
    x_val, y_val = d['x_val'], d['y_val']
    x_test, y_test = d['x_test'], d['y_test']

    cfg['model']['input_dim'] = int(x_train.shape[1])
    model = DARTAGIL(
        input_dim=cfg['model']['input_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        layers=cfg['model']['rnn_layers'],
        dropout=cfg['model']['dropout'],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    tr_loader = DataLoader(FlowDataset(x_train, y_train), batch_size=cfg['training']['batch_size'], shuffle=True)
    va_loader = DataLoader(FlowDataset(x_val, y_val), batch_size=cfg['training']['batch_size'], shuffle=False)
    te_loader = DataLoader(FlowDataset(x_test, y_test), batch_size=cfg['training']['batch_size'], shuffle=False)

    ckpt = dirs['checkpoints'] / f"{cfg['experiment_name']}_{args.experiment}_best.pt"
    history = train(model, tr_loader, va_loader, cfg, device, ckpt, logger)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model_state'])

    if cfg['online_transfer']['enabled'] and args.experiment != 'no_online_tl':
        w = cfg['online_transfer']['window_len']
        if len(x_test) > w and len(x_train) > w:
            loss = online_update(
                model,
                torch.tensor(x_test[:w], dtype=torch.float32),
                torch.tensor(y_test[:w], dtype=torch.long),
                torch.tensor(x_train[-w:], dtype=torch.float32),
                torch.tensor(y_train[-w:], dtype=torch.long),
                lr=cfg['online_transfer']['update_lr'],
                replay_weight=cfg['online_transfer']['replay_weight'],
                device=device,
            )
            logger.info('online_update_loss=%.6f', loss)

    test_stats = evaluate_model(model, te_loader, device)
    metrics = {
        'history': history,
        'test_summary': {'loss': test_stats.loss, 'f1': test_stats.f1, 'auc': test_stats.auc},
        'ablation': args.experiment,
        'result_status': 'Result pending real execution.'
    }

    out_metrics = dirs['results'] / f"{cfg['experiment_name']}_{args.experiment}_train_metrics.json"
    with open(out_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info('Saved %s', out_metrics)


if __name__ == '__main__':
    main()
