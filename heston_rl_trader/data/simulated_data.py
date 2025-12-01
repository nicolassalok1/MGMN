# data/simulated_data.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SimMarketConfig:
    n_steps: int = 10_000
    dt: float = 1.0
    mu_btc: float = 0.0
    mu_shit: float = 0.0
    vol_btc: float = 0.02
    vol_shit: float = 0.05
    seed: int = 123


@dataclass
class SimMarketData:
    btc_prices: np.ndarray
    btc_fut_prices: np.ndarray
    btc_funding: np.ndarray
    btc_oi: np.ndarray
    btc_iv_surface: np.ndarray    # [n_steps, NK, NT]
    k_grid: np.ndarray
    t_grid: np.ndarray

    shit_prices: np.ndarray
    shit_volumes: np.ndarray
    shit_funding: np.ndarray


def simulate_market(config: SimMarketConfig) -> SimMarketData:
    rng = np.random.default_rng(config.seed)
    n = config.n_steps

    # BTC: random walk
    btc_prices = np.zeros(n, dtype=np.float32)
    btc_prices[0] = 30_000.0
    for t in range(1, n):
        z = rng.normal()
        btc_prices[t] = btc_prices[t-1] * np.exp(
            (config.mu_btc - 0.5 * config.vol_btc**2) * config.dt
            + config.vol_btc * np.sqrt(config.dt) * z
        )

    # shitcoin: plus volatile
    shit_prices = np.zeros(n, dtype=np.float32)
    shit_prices[0] = 1.0
    for t in range(1, n):
        z = rng.normal()
        shit_prices[t] = shit_prices[t-1] * np.exp(
            (config.mu_shit - 0.5 * config.vol_shit**2) * config.dt
            + config.vol_shit * np.sqrt(config.dt) * z
        )

    # volumes / funding simples
    shit_volumes = rng.lognormal(mean=0.0, sigma=1.0, size=n).astype(np.float32)
    shit_funding = rng.normal(loc=0.0, scale=0.0005, size=n).astype(np.float32)

    btc_fut_prices = btc_prices * (1.0 + 0.001 * rng.normal(size=n))
    btc_funding = rng.normal(loc=0.0, scale=0.0002, size=n).astype(np.float32)
    btc_oi = rng.lognormal(mean=10.0, sigma=0.3, size=n).astype(np.float32)

    # IV surface toy: ATM IV ~ vol_btc, smile artificiel
    NK, NT = 5, 4
    k_grid = np.linspace(-0.4, 0.4, NK, dtype=np.float32)
    t_grid = np.array([0.05, 0.25, 0.5, 1.0], dtype=np.float32)
    btc_iv_surface = np.zeros((n, NK, NT), dtype=np.float32)

    base_iv = config.vol_btc
    for t in range(n):
        for i, k in enumerate(k_grid):
            for j, T in enumerate(t_grid):
                # smile simple: plus de vol OTM, dépendance T
                smile = 0.2 * abs(k)
                term  = 0.1 * np.log1p(T)
                btc_iv_surface[t, i, j] = base_iv + smile + term

    return SimMarketData(
        btc_prices=btc_prices,
        btc_fut_prices=btc_fut_prices,
        btc_funding=btc_funding,
        btc_oi=btc_oi,
        btc_iv_surface=btc_iv_surface,
        k_grid=k_grid,
        t_grid=t_grid,
        shit_prices=shit_prices,
        shit_volumes=shit_volumes,
        shit_funding=shit_funding,
    )
