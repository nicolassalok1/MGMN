from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyo.lstm_forecaster import LSTMForecaster, build_feature_window  # noqa: E402


def test_lstm_predict_outputs_present():
    cfg = {"use_lstm": True, "lstm_model_path": str(ROOT / "models" / "lstm" / "model.pt")}
    forecaster = LSTMForecaster(cfg)
    features = build_feature_window(np.linspace(1, 2, num=16), features=("returns", "price"), window=8)
    outputs = forecaster.predict(features, horizon=4)
    for key in ("next_return", "vol_forecast", "prob_up"):
        assert key in outputs
