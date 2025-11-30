from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyo.config import PolyoConfig  # noqa: E402
from polyo.rl import LSTM_FEATURE_KEYS, build_obs_with_lstm  # noqa: E402


def test_observation_grows_with_lstm_features():
    prices = np.linspace(1.0, 2.0, num=12)
    base_features = {"price": prices[-1], "vol": 0.1, "regime": 1.0}
    base_order = ("price", "vol", "regime")

    cfg_off = PolyoConfig(use_lstm=False)
    obs_off, names_off = build_obs_with_lstm(
        base_features=base_features,
        base_order=base_order,
        prices=prices,
        config=cfg_off,
        forecaster=None,
    )

    cfg_on = PolyoConfig(use_lstm=True, lstm_horizon=4, lstm_features=["returns", "price"])
    obs_on, names_on = build_obs_with_lstm(
        base_features=base_features,
        base_order=base_order,
        prices=prices,
        config=cfg_on,
        forecaster=None,
    )

    assert obs_on.shape[-1] > obs_off.shape[-1]
    assert len(names_on) - len(names_off) == len(LSTM_FEATURE_KEYS)
