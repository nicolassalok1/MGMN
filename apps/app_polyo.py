from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

# --------------------------------------------------------------------
# sys.path bootstrap for local modules
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rough_bergomi"))
sys.path.insert(0, str(ROOT / "jumpdiff"))
sys.path.insert(0, str(ROOT / "pykalman"))
sys.path.insert(0, str(ROOT / "hmmlearn" / "src"))
sys.path.insert(0, str(ROOT / "RLTrader"))
sys.path.insert(0, str(ROOT / "TradeMaster"))
sys.path.insert(0, str(ROOT / "limit-order-book" / "python"))
sys.path.insert(0, str(ROOT / "Calibrating-Rough-Volatility-Models-with-Deep-Learning"))
sys.path.insert(0, str(ROOT.parent / "src"))

from polyo.config import PolyoConfig
from polyo.lstm_forecaster import get_or_init_lstm_forecaster
from polyo.rl import build_obs_with_lstm


# Utility wrappers with safe imports
def simulate_rough_vol(h: float, eta: float, n: int) -> np.ndarray:
    try:
        import rbergomi  # type: ignore

        if hasattr(rbergomi, "fbm"):
            path = np.array(rbergomi.fbm(h, n)) * eta  # type: ignore[attr-defined]
        else:
            path = np.random.normal(0, eta * 0.1, size=n)
    except Exception:
        path = np.random.normal(0, eta * 0.1, size=n)
    price = 100 + np.cumsum(path)
    return price


def jumpdiff_estimate(length: int, lam: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    base = np.random.normal(0, 0.01, size=length)
    jumps = np.random.poisson(lam=lam, size=length) * np.random.laplace(0, 0.5, size=length)
    returns = base + jumps
    params: Dict[str, Any]
    try:
        import jumpdiff  # type: ignore

        if hasattr(jumpdiff, "estimate"):
            params = jumpdiff.estimate(returns)  # type: ignore[attr-defined]
        else:
            params = {"mean": float(np.mean(returns)), "std": float(np.std(returns))}
    except Exception:
        params = {"mean": float(np.mean(returns)), "std": float(np.std(returns))}
    return returns, params


def kalman_smooth_series(noise: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.cumsum(np.random.normal(0, noise, size=n))
    try:
        from pykalman import KalmanFilter  # type: ignore

        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            transition_covariance=[noise**2],
            observation_covariance=[noise**2],
        )
        smoothed, _ = kf.smooth(data.reshape(-1, 1))
        return data, smoothed[:, 0]
    except Exception:
        smoothed = np.convolve(data, np.ones(5) / 5, mode="same")
        return data, smoothed


def hmm_regimes(n_states: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.column_stack(
        [
            np.random.normal(0, 1, size=n),
            np.random.laplace(0, 2, size=n),
        ]
    )
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=20, random_state=42)
        model.fit(feats)
        regimes = model.predict(feats)
    except Exception:
        regimes = np.zeros(n, dtype=int)
    return feats, regimes


# Minimal RL env
@dataclass
class MiniEnv:
    n: int
    prices: np.ndarray
    vols: np.ndarray
    config: PolyoConfig
    forecaster: Any | None = None
    idx: int = 0
    pos: int = 0  # -1,0,1
    base_order: Tuple[str, ...] = ("price", "vol", "position")

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.pos = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        prev_pos = self.pos
        self.pos = {0: 0, 1: 1, 2: -1}.get(action, 0)
        reward = 0.0
        if self.idx < self.n - 1:
            ret = self.prices[self.idx + 1] - self.prices[self.idx]
            reward = ret * prev_pos
        self.idx += 1
        done = self.idx >= self.n - 1
        return self._obs(), float(reward), done

    def _obs(self) -> np.ndarray:
        i = min(self.idx, self.n - 1)
        base_features = {
            "price": self.prices[i],
            "vol": self.vols[i],
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


def rl_demo(steps: int, config: PolyoConfig) -> List[Dict[str, Any]]:
    prices = 100 + np.cumsum(np.random.normal(0, 1, size=steps))
    vols = np.abs(np.diff(np.concatenate([[prices[0]], prices]))) + 1e-4
    forecaster = get_or_init_lstm_forecaster(config)
    env = MiniEnv(n=steps, prices=prices, vols=vols, config=config, forecaster=forecaster)
    obs = env.reset()
    traj: List[Dict[str, Any]] = []
    for _ in range(steps):
        action = np.random.choice([0, 1, 2])
        obs, reward, done = env.step(int(action))
        traj.append({"obs": obs.tolist(), "action": int(action), "reward": reward})
        if done:
            break
    return traj


# --------------------------------------------------------------------
# Streamlit UI
st.set_page_config(page_title="POLYO Demo", layout="wide")
st.title("POLYO Modules Playground")
st.write("Interact with rough volatility, jump-diffusion, Kalman smoothing, HMM regimes, and a mini RL demo.")

section = st.sidebar.radio(
    "Sections",
    [
        "Rough Volatility Simulator",
        "Jump Diffusion",
        "Kalman Filter",
        "Regime Detection (HMM)",
        "RL Mini-Demo",
    ],
)

# Rough Volatility
if section == "Rough Volatility Simulator":
    st.header("Rough Volatility Simulator")
    H = st.sidebar.slider("H (Hurst)", 0.01, 0.5, 0.1, 0.01)
    eta = st.sidebar.slider("eta", 0.1, 3.0, 0.5, 0.1)
    n = st.sidebar.slider("Number of steps", 16, 256, 64, 8)
    if st.button("Simulate Rough Vol Path"):
        series = simulate_rough_vol(H, eta, n)
        st.line_chart(series)
        st.write({"mean": float(np.mean(series)), "std": float(np.std(series))})

# Jump Diffusion
elif section == "Jump Diffusion":
    st.header("Jump Diffusion")
    length = st.sidebar.slider("Series length", 16, 512, 128, 16)
    lam = st.sidebar.slider("Jump intensity (lambda)", 0.0, 10.0, 2.0, 0.5)
    if st.button("Generate + Estimate Jumps"):
        returns, params = jumpdiff_estimate(length, lam)
        st.line_chart(returns)
        st.write(params)

# Kalman Filter
elif section == "Kalman Filter":
    st.header("Kalman Filter Smoother")
    noise = st.sidebar.slider("Noise level", 0.01, 2.0, 0.2, 0.01)
    n = st.sidebar.slider("Series length", 16, 256, 64, 8)
    if st.button("Smooth Series"):
        raw, smooth = kalman_smooth_series(noise, n)
        st.line_chart({"raw": raw, "smoothed": smooth})
        st.write({"raw_std": float(np.std(raw)), "smoothed_std": float(np.std(smooth))})

# HMM Regime Detection
elif section == "Regime Detection (HMM)":
    st.header("HMM Regime Detection")
    n_states = st.sidebar.selectbox("Number of regimes", [2, 3], index=0)
    n = st.sidebar.slider("Series length", 16, 256, 64, 8)
    if st.button("Fit HMM"):
        feats, regimes = hmm_regimes(n_states, n)
        st.line_chart(regimes)
        counts = {int(k): int(v) for k, v in zip(*np.unique(regimes, return_counts=True))}
        st.write({"counts": counts})

# RL Mini-Demo
elif section == "RL Mini-Demo":
    st.header("RL Mini-Demo (random policy)")
    steps = st.sidebar.slider("Number of steps", 5, 20, 10, 1)
    use_lstm = st.sidebar.checkbox("Activer LSTM", value=False)
    lstm_horizon = st.sidebar.slider("Horizon LSTM (pas)", 1, 32, 8, 1)
    lstm_features = st.sidebar.multiselect(
        "LSTM features",
        options=["returns", "price", "volatility"],
        default=["returns", "price"],
    )
    lstm_model_path = st.sidebar.text_input("Chemin du modèle LSTM", value="models/lstm/model.pt")
    config = PolyoConfig(
        use_lstm=use_lstm,
        lstm_horizon=lstm_horizon,
        lstm_features=lstm_features or ["returns"],
        lstm_model_path=lstm_model_path,
    )

    if st.button("Run RL Demo"):
        traj = rl_demo(steps, config=config)
        st.write(traj)
        rewards = [t["reward"] for t in traj]
        st.line_chart(rewards)
