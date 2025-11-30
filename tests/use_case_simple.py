from __future__ import annotations

"""
Lightweight end-to-end demo across POLYO submodules.
Steps: rough path -> returns -> jumpdiff params -> Kalman smoothing -> HMM regimes -> tiny RL env rollout.
"""

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}")


# -------------------------------------------------------------------
# sys.path bootstrap (mimics local editable layout)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rough_bergomi"))
sys.path.insert(0, str(ROOT / "jumpdiff"))
sys.path.insert(0, str(ROOT / "pykalman"))
sys.path.insert(0, str(ROOT / "hmmlearn" / "src"))
sys.path.insert(0, str(ROOT / "src"))
# -------------------------------------------------------------------

from polyo.config import PolyoConfig
from polyo.lstm_forecaster import get_or_init_lstm_forecaster
from polyo.rl import ObservationBuilder, build_obs_with_lstm


def simulate_prices(n: int = 16) -> np.ndarray:
    """Small rough-ish path; fallback to Gaussian if rbergomi missing."""
    try:
        import rbergomi  # type: ignore

        if hasattr(rbergomi, "fbm"):
            noise = np.array(rbergomi.fbm(0.1, n)) * 0.2  # type: ignore[attr-defined]
        else:
            noise = np.random.normal(0, 0.02, size=n)
    except Exception:
        noise = np.random.normal(0, 0.02, size=n)
    price = 100 + np.cumsum(noise)
    return price


def jumpdiff_estimate(returns: np.ndarray) -> Dict[str, Any]:
    try:
        import jumpdiff  # type: ignore

        if hasattr(jumpdiff, "estimate"):
            params = jumpdiff.estimate(returns)  # type: ignore[attr-defined]
        else:
            params = {"mean": float(np.mean(returns)), "std": float(np.std(returns))}
    except Exception:
        params = {"mean": float(np.mean(returns)), "std": float(np.std(returns))}
    return params


def kalman_smooth(data: np.ndarray) -> np.ndarray:
    try:
        from pykalman import KalmanFilter  # type: ignore

        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=[0.01],
            observation_covariance=[0.05],
        )
        smoothed, _ = kf.smooth(data.reshape(-1, 1))
        return smoothed[:, 0]
    except Exception:
        return np.convolve(data, np.ones(3) / 3, mode="same")


def hmm_classify(features: np.ndarray) -> np.ndarray:
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=15, random_state=0)
        model.fit(features)
        return model.predict(features)
    except Exception:
        return np.zeros(len(features), dtype=int)


@dataclass
class TinyEnv:
    prices: np.ndarray
    vols: np.ndarray
    regimes: np.ndarray
    config: PolyoConfig
    forecaster: Any | None = None
    idx: int = 0
    position: int = 0  # -1,0,1
    base_order: Tuple[str, ...] = field(default_factory=lambda: ("price", "vol", "regime", "position"))

    def __post_init__(self):
        self.obs_builder = ObservationBuilder(self.base_order)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.idx = 0
        self.position = 0
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_pos = self.position
        self.position = {0: 0, 1: 1, 2: -1}.get(action, 0)
        reward = 0.0
        if self.idx < len(self.prices) - 1:
            ret = self.prices[self.idx + 1] - self.prices[self.idx]
            reward = ret * prev_pos
        self.idx += 1
        done = self.idx >= len(self.prices) - 1
        return self._obs(), float(reward), done, False, {}

    def _obs(self) -> np.ndarray:
        i = min(self.idx, len(self.prices) - 1)
        base_features = {
            "price": self.prices[i],
            "vol": self.vols[i],
            "regime": float(self.regimes[i]),
            "position": float(self.position),
        }
        obs, _ = build_obs_with_lstm(
            base_features=base_features,
            base_order=self.base_order,
            prices=self.prices[: i + 1],
            config=self.config,
            forecaster=self.forecaster,
        )
        return obs


def run_pipeline():
    prices = simulate_prices(n=16)
    returns = np.diff(prices)
    jump_params = jumpdiff_estimate(returns)
    smoothed = kalman_smooth(returns)
    feats = np.column_stack([smoothed, returns])
    regimes = hmm_classify(feats)

    config = PolyoConfig(use_lstm=True, lstm_horizon=4, lstm_features=["returns", "price"])
    forecaster = get_or_init_lstm_forecaster(config) if config.use_lstm else None
    env = TinyEnv(prices=prices, vols=np.abs(smoothed), regimes=regimes, config=config, forecaster=forecaster)
    obs, _ = env.reset()
    traj = []
    for _ in range(10):
        action = random.randint(0, 2)
        obs, reward, done, _, _ = env.step(action)
        traj.append((obs.tolist(), action, reward))
        if done:
            break

    log("INFO", f"prices(first5)={prices[:5].round(4).tolist()}")
    log("INFO", f"returns(first5)={returns[:5].round(4).tolist()}")
    log("INFO", f"jumpdiff params={jump_params}")
    log("INFO", f"smoothed(first5)={smoothed[:5].round(4).tolist()}")
    log("INFO", f"regimes(first10)={regimes[:10].tolist()}")
    log("INFO", f"obs_dim={len(obs)} lstm={'on' if config.use_lstm else 'off'}")
    for idx, step in enumerate(traj):
        log("INFO", f"rl_step[{idx}] obs={step[0]} action={step[1]} reward={step[2]}")
    log("SUCCESS", "use_case_simple completed")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
