from __future__ import annotations

import argparse
import numpy as np

from polyo.lstm_forecaster import LSTMForecaster, build_feature_window


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick LSTM forecast using saved artifacts.")
    parser.add_argument("--config", type=str, default="config/lstm.yaml", help="Path to LSTM YAML config.")
    parser.add_argument("--horizon", type=int, default=8, help="Forecast horizon.")
    args = parser.parse_args()

    forecaster = LSTMForecaster.from_config(args.config)
    # Tiny synthetic series for demonstration
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, size=64))
    features = build_feature_window(prices, features=("returns", "price"), window=args.horizon * 2)
    forecast = forecaster.predict(features, horizon=args.horizon)
    print(forecast)  # noqa: T201


if __name__ == "__main__":
    main()
