from __future__ import annotations

import torch
from torch import nn


def online_update(model, x_new, y_new, x_replay, y_replay, lr: float, replay_weight: float, device: torch.device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    x_new, y_new = x_new.to(device), y_new.to(device)
    x_replay, y_replay = x_replay.to(device), y_replay.to(device)

    logits_new, _, _, _ = model(x_new)
    logits_replay, _, _, _ = model(x_replay)

    loss = ce(logits_new, y_new) + replay_weight * ce(logits_replay, y_replay)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())
