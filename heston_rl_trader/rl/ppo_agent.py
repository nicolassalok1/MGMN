# rl/ppo_agent.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOAgent:
    def __init__(self, obs_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.pi_net = MLP(obs_dim, 64).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, 1, device=self.device))
        self.v_net = MLP(obs_dim, 1).to(self.device)

        self.pi_optimizer = optim.Adam(list(self.pi_net.parameters()) + [self.log_std], lr=cfg.pi_lr)
        self.vf_optimizer = optim.Adam(self.v_net.parameters(), lr=cfg.vf_lr)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        self.pi_net.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self._distribution(obs_t)
        a = dist.sample()
        logp = dist.log_prob(a).sum(axis=-1)
        self.pi_net.train()
        return a.cpu().numpy()[0], float(logp.item()), float(dist.mean.cpu().detach().numpy()[0])

    def compute_logp(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        return dist.log_prob(act).sum(axis=-1)

    def update(self, buf: Dict[str, np.ndarray]):
        obs = torch.as_tensor(buf["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(buf["act"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(buf["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(buf["ret"], dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(buf["logp"], dtype=torch.float32, device=self.device)

        # normalisation des avantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Policy update
        for _ in range(self.cfg.train_pi_iters):
            logp = self.compute_logp(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.pi_net.parameters(), self.cfg.max_grad_norm)
            self.pi_optimizer.step()

            kl = (logp_old - logp).mean().item()
            if kl > 1.5 * self.cfg.target_kl:
                break

        # Value update
        for _ in range(self.cfg.train_v_iters):
            v = self.v_net(obs).squeeze(-1)
            loss_v = ((v - ret) ** 2).mean()
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), self.cfg.max_grad_norm)
            self.vf_optimizer.step()


def compute_gae(rews, vals, dones, gamma, lam):
    """
    GAE-Lambda pour avantage et retours.
    """
    adv = np.zeros_like(rews, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rews))):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * vals[t + 1] * nonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals[:-1]
    return adv, ret
