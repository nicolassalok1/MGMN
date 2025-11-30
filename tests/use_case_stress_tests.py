from __future__ import annotations

"""
Stress scenarios exercising POLYO submodules. Each scenario is lightweight and guarded to avoid crashes.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

# sys.path bootstrap
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rough_bergomi"))
sys.path.insert(0, str(ROOT / "jumpdiff"))
sys.path.insert(0, str(ROOT / "pykalman"))
sys.path.insert(0, str(ROOT / "hmmlearn" / "src"))
sys.path.insert(0, str(ROOT / "limit-order-book" / "python"))
sys.path.insert(0, str(ROOT / "src"))

from polyo.config import PolyoConfig
from polyo.lstm_forecaster import get_or_init_lstm_forecaster
from polyo.rl import build_obs_with_lstm


def log(level: str, message: str):
    print(f"[{level}] {message}")


# Scenario A — Rough Volatility Stress
def scenario_rough_vol():
    try:
        import rbergomi  # type: ignore

        H = 0.01
        N = 512
        eta = 3.5
        if hasattr(rbergomi, "fbm"):
            noise = np.array(rbergomi.fbm(H, N)) * eta  # type: ignore[attr-defined]
        else:
            noise = np.random.normal(0, eta * 0.1, size=N)
    except Exception as exc:
        log("WARNING", f"rough_vol fallback due to {exc}")
        noise = np.random.normal(0, 0.4, size=512)

    stats = {
        "max": float(np.max(noise)),
        "min": float(np.min(noise)),
        "kurtosis": float(np.mean((noise - noise.mean()) ** 4) / (np.var(noise) ** 2 + 1e-9)),
        "mean": float(np.mean(noise)),
    }
    log("INFO", f"rough_vol.stats {stats}")


# Scenario B — Jump-Diffusion Explosion Stress
def scenario_jumpdiff():
    lam = 6.0
    n = 256
    base = np.random.normal(0, 0.01, size=n)
    jumps = np.random.poisson(lam=lam, size=n) * np.random.laplace(0, 0.5, size=n)
    returns = base + jumps
    try:
        import jumpdiff  # type: ignore

        if hasattr(jumpdiff, "estimate"):
            params = jumpdiff.estimate(returns)  # type: ignore[attr-defined]
        else:
            params = {"mean": float(np.mean(returns)), "std": float(np.std(returns))}
    except Exception as exc:
        log("WARNING", f"jumpdiff fallback due to {exc}")
        params = {"mean": float(np.mean(returns)), "std": float(np.std(returns))}
    detected = int(np.sum(np.abs(returns) > 0.5))
    log("INFO", f"jumpdiff.stats {{'detected_jumps': {detected}, 'params': {params}}}")


# Scenario C — Microstructure Noise Stress
def scenario_orderbook():
    try:
        import limitorderbook as lob  # type: ignore
    except Exception:
        try:
            import olob as lob  # type: ignore
        except Exception as exc:
            log("WARNING", f"orderbook bindings unavailable ({exc}); skipping")
            return
    book = lob.OrderBook() if hasattr(lob, "OrderBook") else None
    if not book:
        log("WARNING", "OrderBook class missing; skipping")
        return
    for _ in range(5):
        price = 100 + random.uniform(-5, 5)
        size = random.uniform(0.1, 10)
        if random.random() < 0.5:
            book.insert("bid", price, size)
        else:
            book.insert("ask", price, size)
    try:
        top = {"best_bid": book.best_bid(), "best_ask": book.best_ask()}
    except Exception as exc:
        log("ERROR", f"top-of-book failed ({exc})")
        return
    spread = top["best_ask"][0] - top["best_bid"][0] if top["best_bid"] and top["best_ask"] else math.nan
    log("INFO", f"lob.top {{'top': {top}, 'spread': {spread}}}")


# Scenario D — Kalman Filter Stress
def scenario_kalman():
    data = np.random.normal(0, 1, size=128)
    spikes_idx = np.random.choice(len(data), size=5, replace=False)
    data[spikes_idx] += np.random.normal(0, 10, size=5)
    try:
        from pykalman import KalmanFilter  # type: ignore

        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=[1.0],
            observation_covariance=[2.0],
        )
        smoothed, _ = kf.smooth(data.reshape(-1, 1))
        out = smoothed[:, 0]
    except Exception as exc:
        log("WARNING", f"kalman fallback mean due to {exc}")
        out = np.convolve(data, np.ones(5) / 5, mode="same")
    stats = {"input_std": float(np.std(data)), "output_std": float(np.std(out))}
    log("INFO", f"kalman.stats {stats}")


# Scenario E — Regime Detection Stress
def scenario_hmm():
    feats = np.column_stack(
        [
            np.random.normal(0, 1, size=128),
            np.random.laplace(0, 2, size=128),
        ]
    )
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=20, random_state=42)
        model.fit(feats)
        regimes = model.predict(feats)
        switch_freq = int(np.sum(regimes[:-1] != regimes[1:]))
        log("INFO", f"hmm.stats {{'switches': {switch_freq}, 'unique_regimes': {int(len(set(regimes)))} }}")
    except Exception as exc:
        log("WARNING", f"hmm fallback due to {exc}")
        log("INFO", "hmm.stats {'switches': 0, 'unique_regimes': 1}")


# Scenario F — RL Environment Stress
class StressEnv:
    def __init__(self, config: PolyoConfig, forecaster=None):
        self.n = 12
        self.prices = 100 + np.cumsum(np.random.normal(0, 3, size=self.n))
        self.regimes = np.random.randint(0, 3, size=self.n)
        self.liquidity = np.random.choice([1.0, 0.5, 0.1], size=self.n, p=[0.5, 0.3, 0.2])
        self.idx = 0
        self.pos = 0
        self.config = config
        self.forecaster = forecaster
        self.base_order = ("price", "regime", "liquidity", "position")

    def reset(self):
        self.idx = 0
        self.pos = 0
        return self._obs(), {}

    def step(self, action: int):
        prev_pos = self.pos
        self.pos = {0: 0, 1: 1, 2: -1}.get(action, 0)
        reward = 0.0
        if self.idx < self.n - 1:
            ret = self.prices[self.idx + 1] - self.prices[self.idx]
            reward = ret * prev_pos * self.liquidity[self.idx]
        self.idx += 1
        done = self.idx >= self.n - 1
        return self._obs(), float(reward), done, False, {"liquidity": float(self.liquidity[self.idx - 1])}

    def _obs(self):
        i = min(self.idx, self.n - 1)
        base_features = {
            "price": self.prices[i],
            "regime": float(self.regimes[i]),
            "liquidity": float(self.liquidity[i]),
            "position": float(self.pos),
        }
        obs, _ = build_obs_with_lstm(
            base_features=base_features,
            base_order=self.base_order,
            prices=self.prices[: i + 1],
            config=self.config,
            forecaster=self.forecaster,
        )
        return obs


def scenario_rl():
    config = PolyoConfig(use_lstm=True, lstm_horizon=4, lstm_features=["returns", "price"])
    forecaster = get_or_init_lstm_forecaster(config)
    env = StressEnv(config=config, forecaster=forecaster)
    obs, _ = env.reset()
    traj: List[Any] = []
    for _ in range(10):
        action = random.randint(0, 2)
        obs, reward, done, _, info = env.step(action)
        traj.append({"obs": obs.tolist(), "action": action, "reward": reward, "liquidity": info["liquidity"]})
        if done:
            break
    log("INFO", f"rl.trajectory {traj}")


def main():
    scenario_rough_vol()
    scenario_jumpdiff()
    scenario_orderbook()
    scenario_kalman()
    scenario_hmm()
    scenario_rl()
    log("SUCCESS", "stress tests completed")


if __name__ == "__main__":
    main()
