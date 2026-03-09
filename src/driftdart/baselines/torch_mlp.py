from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers.extend([nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)])
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLPBaseline:
    def __init__(self, name: str, hidden: tuple[int, ...], dropout: float):
        self.name = name
        self.hidden = hidden
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1e-3):
        self.model = _MLP(x_train.shape[1], self.hidden, self.dropout).to(self.device)
        ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = ce(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(x, dtype=torch.float32, device=self.device)
            p = torch.softmax(self.model(xb), dim=1)[:, 1].cpu().numpy()
        return p

    def update(self, x_new: np.ndarray, y_new: np.ndarray, steps: int = 20, lr: float = 1e-4):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()
        xb = torch.tensor(x_new, dtype=torch.float32, device=self.device)
        yb = torch.tensor(y_new, dtype=torch.long, device=self.device)
        for _ in range(steps):
            logits = self.model(xb)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
