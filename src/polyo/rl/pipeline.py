from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from polyo.config import PolyoConfig
from polyo.lstm_forecaster import build_feature_window, get_or_init_lstm_forecaster
from polyo.rl.observation import build_observation_vector


def compute_lstm_outputs_from_prices(
    prices: np.ndarray,
    config: PolyoConfig,
    forecaster=None,
) -> Optional[Dict[str, float]]:
    """
    Helper that prepares past features and queries the LSTM forecaster.
    Uses only past prices to avoid any data leakage.
    """

    if not config.lstm_enabled:
        return None

    forecaster = forecaster or get_or_init_lstm_forecaster(config)
    if forecaster is None:
        return None

    prices = np.asarray(prices, dtype=float)
    window = max(config.lstm_horizon * 2, 4)
    feats = build_feature_window(
        prices=prices,
        features=config.lstm_features,
        window=window,
    )
    return forecaster.predict(feats, horizon=config.lstm_horizon)


def build_obs_with_lstm(
    base_features: Dict[str, float],
    base_order: Sequence[str],
    prices: np.ndarray,
    config: PolyoConfig,
    forecaster=None,
) -> Tuple[np.ndarray, Sequence[str]]:
    """
    Single entry point for observation construction + LSTM augmentation.
    """

    lstm_outputs = compute_lstm_outputs_from_prices(prices, config=config, forecaster=forecaster)
    return build_observation_vector(
        base_features=base_features,
        base_order=base_order,
        lstm_outputs=lstm_outputs,
    )
