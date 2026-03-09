from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from driftdart.models.dart_agil import DARTAGIL, fgsm_like_perturbation, kl_divergence


@dataclass
class EpochStats:
    loss: float
    f1: float
    auc: float


def _compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    y_pred = (prob >= thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = float("nan")
    return {"f1": float(f1), "auc": float(auc)}


def evaluate_model(model: DARTAGIL, loader: DataLoader, device: torch.device) -> EpochStats:
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses = []
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, mu, logvar, _ = model(x)
            loss = ce(logits, y) + 0.1 * kl_divergence(mu, logvar)
            losses.append(loss.item())
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            ys.append(y.cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(probs) if probs else np.array([])
    m = _compute_metrics(y_true, p) if len(y_true) else {"f1": 0.0, "auc": float("nan")}
    return EpochStats(loss=float(np.mean(losses) if losses else 0.0), f1=m["f1"], auc=m["auc"])


def train(
    model: DARTAGIL,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    logger,
):
    tcfg = cfg["training"]
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0
    patience = 0
    history = []

    for epoch in range(tcfg["epochs"]):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits, mu, logvar, _ = model(x)
            base_loss = ce(logits, y)
            kl = kl_divergence(mu, logvar)

            if cfg.get("ablation") == "no_agil":
                adv_loss = torch.tensor(0.0, device=device)
            else:
                x_adv = fgsm_like_perturbation(model, x, y, eps=tcfg["adv_eps"])
                logits_adv, _, _, _ = model(x_adv)
                adv_loss = ce(logits_adv, y)

            if cfg.get("ablation") == "no_dart":
                kl = torch.tensor(0.0, device=device)

            loss = base_loss + tcfg["kl_weight"] * kl + tcfg["agil_weight"] * adv_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            batch_losses.append(loss.item())

        val_stats = evaluate_model(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)), "val_loss": val_stats.loss, "val_f1": val_stats.f1, "val_auc": val_stats.auc})
        logger.info("epoch=%d train_loss=%.4f val_f1=%.4f val_auc=%.4f", epoch + 1, np.mean(batch_losses), val_stats.f1, val_stats.auc)

        if val_stats.f1 > best_f1:
            best_f1 = val_stats.f1
            patience = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, checkpoint_path)
        else:
            patience += 1

        if patience >= tcfg["early_stopping_patience"]:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    return history
