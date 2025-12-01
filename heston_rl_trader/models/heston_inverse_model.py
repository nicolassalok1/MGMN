# models/heston_inverse_model.py
from __future__ import annotations
import torch
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, nk: int, nt: int, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        nk4 = nk // 4
        nt4 = nt // 4
        self.flat_dim = 64 * nk4 * nt4
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        z = self.fc(h)
        return z


class ParamHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SurfaceHead(nn.Module):
    def __init__(self, in_dim: int, nk: int, nt: int):
        super().__init__()
        self.nk = nk
        self.nt = nt
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, nk * nt),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, self.nk, self.nt)


class InverseHestonModel(nn.Module):
    """
    Modèle inverse Heston:
      - Input: surface (ou pseudo-surface) [B,1,NK,NT]
      - Output:
         params_scaled [B,5] (kappa, theta, sigma, rho, v0 en espace normalisé)
         surface_recon [B,1,NK,NT] (reconstruction pour loss de surface)
    """

    def __init__(self, nk: int, nt: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = EncoderCNN(nk, nt, hidden_dim)
        self.head_kappa = ParamHead(hidden_dim, 1)
        self.head_theta = ParamHead(hidden_dim, 1)
        self.head_sigma = ParamHead(hidden_dim, 1)
        self.head_rho   = ParamHead(hidden_dim, 1)
        self.head_v0    = ParamHead(hidden_dim, 1)
        self.surface_head = SurfaceHead(hidden_dim, nk, nt)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        kappa_s = self.head_kappa(z)
        theta_s = self.head_theta(z)
        sigma_s = self.head_sigma(z)
        rho_s   = self.head_rho(z)
        v0_s    = self.head_v0(z)
        params = torch.cat([kappa_s, theta_s, sigma_s, rho_s, v0_s], dim=-1)
        surf_recon = self.surface_head(z)
        return params, surf_recon


class DummyHestonInverse(nn.Module):
    """
    Fallback simple: renvoie des zéros.
    Utile pour lancer le projet avant d'avoir le vrai modèle entraîné.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        params = torch.zeros(B, 5, device=x.device, dtype=torch.float32)
        surf_recon = torch.zeros_like(x)
        return params, surf_recon


def load_heston_inverse_model(
    nk: int,
    nt: int,
    ckpt_path: str | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Charge un InverseHestonModel avec ou sans checkpoint.
    Si ckpt_path est None, renvoie un DummyHestonInverse.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path is None:
        model = DummyHestonInverse().to(device)
        model.eval()
        return model

    model = InverseHestonModel(nk, nt, hidden_dim=128).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model
