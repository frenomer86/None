from __future__ import annotations

import numpy as np
import torch
from torch import nn


def run_few_shot(model, x_train, y_train, x_test, y_test, n: int, repeats: int, device: torch.device):
    rng = np.random.default_rng(42)
    f1s = []
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(repeats):
        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]
        if len(idx_pos) < n or len(idx_neg) < n:
            continue
        idx = np.concatenate([rng.choice(idx_pos, n, replace=False), rng.choice(idx_neg, n, replace=False)])

        model.load_state_dict(base_state)
        model.to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        ce = nn.CrossEntropyLoss()

        xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
        yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
        for _ in range(10):
            logits, _, _, _ = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x_test, dtype=torch.float32, device=device)
            prob = torch.softmax(model(xt)[0], dim=1)[:, 1].cpu().numpy()
        from sklearn.metrics import f1_score
        f1s.append(f1_score(y_test, (prob >= 0.5).astype(int), zero_division=0))

    if not f1s:
        return {"mean_f1": None, "std_f1": None, "n": n, "repeats": repeats}
    return {"mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)), "n": n, "repeats": len(f1s)}
