from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from driftdart.data.dataset import FlowDataset
from driftdart.models.dart_agil import DARTAGIL
from driftdart.training.engine import train
from driftdart.training.online_transfer import online_update


class DARTAGILWrapper:
    def __init__(self, cfg: dict, name: str = "dart_agil"):
        self.name = name
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
        self.model = DARTAGIL(
            input_dim=x_train.shape[1],
            hidden_dim=self.cfg['model']['hidden_dim'],
            latent_dim=self.cfg['model']['latent_dim'],
            layers=self.cfg['model']['rnn_layers'],
            dropout=self.cfg['model']['dropout'],
        ).to(self.device)

        tr_loader = DataLoader(FlowDataset(x_train, y_train), batch_size=self.cfg['training']['batch_size'], shuffle=True)
        va_loader = DataLoader(FlowDataset(x_val, y_val), batch_size=self.cfg['training']['batch_size'], shuffle=False)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / 'tmp_best.pt'
            class _Logger:
                def info(self, *args, **kwargs):
                    pass
            train(self.model, tr_loader, va_loader, self.cfg, self.device, ckpt, _Logger())
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state['model_state'])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(x, dtype=torch.float32, device=self.device)
            logits, _, _, _ = self.model(xb)
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    def update(self, x_new: np.ndarray, y_new: np.ndarray, x_replay: np.ndarray, y_replay: np.ndarray):
        online_update(
            self.model,
            torch.tensor(x_new, dtype=torch.float32),
            torch.tensor(y_new, dtype=torch.long),
            torch.tensor(x_replay, dtype=torch.float32),
            torch.tensor(y_replay, dtype=torch.long),
            lr=self.cfg['online_transfer']['update_lr'],
            replay_weight=self.cfg['online_transfer']['replay_weight'],
            device=self.device,
        )
