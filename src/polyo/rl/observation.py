from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LSTM_FEATURE_KEYS: List[str] = ["next_return", "vol_forecast", "prob_up"]


@dataclass
class ObservationBuilder:
    """
    Small helper that builds an observation vector in a single, documented place.

    Base feature names are provided by the caller to keep environments explicit.
    When LSTM outputs are supplied, they are appended in a fixed order to avoid
    shape mismatches throughout the pipeline.
    """

    base_feature_names: Sequence[str]
    normalise: bool = False

    def build(
        self,
        base_features: Dict[str, float],
        lstm_outputs: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        values: List[float] = []
        names: List[str] = []

        for name in self.base_feature_names:
            raw = float(base_features.get(name, 0.0))
            if not np.isfinite(raw):
                raw = 0.0
            values.append(raw)
            names.append(name)

        if lstm_outputs:
            for key in LSTM_FEATURE_KEYS:
                values.append(float(lstm_outputs.get(key, 0.0)))
                names.append(f"lstm_{key}")

        obs = np.asarray(values, dtype=float)
        if self.normalise and obs.size:
            mu = float(np.mean(obs))
            sigma = float(np.std(obs)) + 1e-6
            obs = (obs - mu) / sigma
        return obs, names


def build_observation_vector(
    base_features: Dict[str, float],
    base_order: Sequence[str],
    lstm_outputs: Optional[Dict[str, float]] = None,
    normalise: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Functional wrapper around ObservationBuilder for quick use cases.
    """

    builder = ObservationBuilder(base_order, normalise=normalise)
    return builder.build(base_features, lstm_outputs)
