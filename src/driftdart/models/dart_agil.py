from __future__ import annotations

import torch
from torch import nn


class DARTEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, layers: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class DARTDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor):
        return self.net(z)


class DARTAGIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, layers: int, dropout: float):
        super().__init__()
        self.encoder = DARTEncoder(input_dim, hidden_dim, latent_dim, layers, dropout)
        self.decoder = DARTDecoder(latent_dim, hidden_dim, dropout)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar, z


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def fgsm_like_perturbation(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    logits, _, _, _ = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    grad = x_adv.grad.detach().sign()
    return torch.clamp(x + eps * grad, 0.0, 1.0).detach()
