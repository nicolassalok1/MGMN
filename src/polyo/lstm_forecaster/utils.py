from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from polyo.config import PolyoConfig


def build_feature_window(
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    features: Sequence[str] = ("returns",),
    window: int = 32,
) -> np.ndarray:
    """
    Build a sliding window matrix for the LSTM forecaster.

    Only past information is used; callers should slice prices to the current
    timestep before passing them here.
    """

    prices = np.asarray(prices, dtype=float)
    if prices.ndim == 0:
        prices = prices.reshape(1)
    start = max(0, prices.shape[0] - window)
    window_prices = prices[start:]
    cols = []

    if "price" in features:
        cols.append(window_prices - window_prices.mean())

    if "returns" in features or "log_returns" in features:
        returns = np.diff(np.log(window_prices + 1e-9))
        if returns.size == 0:
            returns = np.zeros(1, dtype=float)
        cols.append(returns)

    if "volume" in features and volumes is not None:
        volumes = np.asarray(volumes, dtype=float)
        cols.append(volumes[-len(window_prices) :])

    if "volatility" in features:
        if cols:
            rolling_vol = np.std(np.column_stack(cols), axis=1)
        else:
            rolling_vol = np.std(window_prices) * np.ones_like(window_prices)
        cols.append(rolling_vol)

    if not cols:
        cols.append(np.diff(window_prices, prepend=window_prices[0]))

    # Align column lengths
    min_len = min(len(c) for c in cols)
    cols = [c[-min_len:] for c in cols]
    return np.column_stack(cols)


@lru_cache(maxsize=2)
def _cached_forecaster(model_path: str, use_config_path: str) -> "LSTMForecaster | None":
    try:
        from polyo.lstm_forecaster.model import LSTMForecaster
    except Exception:
        return None

    cfg = {
        "lstm_model_path": model_path,
        "lstm_config_path": use_config_path,
    }
    try:
        return LSTMForecaster(cfg)
    except Exception:
        return None


def get_or_init_lstm_forecaster(config: PolyoConfig | Dict) -> "LSTMForecaster | None":
    """
    Return a cached LSTMForecaster instance if LSTM is enabled; otherwise None.
    """

    cfg_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    if not cfg_dict.get("use_lstm"):
        return None

    model_path = cfg_dict.get("lstm_model_path", "models/lstm/model.pt")
    config_path = cfg_dict.get("lstm_config_path", "config/lstm.yaml")
    return _cached_forecaster(model_path, config_path)
