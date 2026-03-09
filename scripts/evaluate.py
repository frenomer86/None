#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from driftdart.utils.config import ensure_dirs, load_config
from driftdart.data.dataset import FlowDataset
from driftdart.models.dart_agil import DARTAGIL
from driftdart.eval.metrics import classification_metrics
from driftdart.eval.fewshot import run_few_shot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--experiment', default='full')
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_dirs(cfg['paths']['output_root'])

    d = np.load(dirs['processed'] / f"{cfg['experiment_name']}.npz")
    x_train, y_train = d['x_train'], d['y_train']
    x_test, y_test = d['x_test'], d['y_test']

    model = DARTAGIL(
        input_dim=x_train.shape[1],
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        layers=cfg['model']['rnn_layers'],
        dropout=cfg['model']['dropout'],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model_state'])
    model.to(device)
    model.eval()

    loader = DataLoader(FlowDataset(x_test, y_test), batch_size=cfg['training']['batch_size'], shuffle=False)
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _, _, _ = model(x)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())

    prob = np.concatenate(probs)
    y_true = np.concatenate(ys)
    test_metrics = classification_metrics(y_true, prob, threshold=cfg['evaluation']['threshold'])

    few_shot = []
    for n in cfg['evaluation']['few_shot_ns']:
        few_shot.append(run_few_shot(model, x_train, y_train, x_test, y_test, n=n, repeats=cfg['evaluation']['few_shot_repeats'], device=device))

    output = {
        'experiment': args.experiment,
        'test': test_metrics,
        'few_shot': few_shot,
        'result_status': 'Result pending real execution.'
    }

    out = dirs['results'] / f"{cfg['experiment_name']}_{args.experiment}_metrics.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"Saved evaluation metrics: {out}")


if __name__ == '__main__':
    main()
