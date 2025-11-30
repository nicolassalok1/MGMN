from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class _TinyLSTM(nn.Module):  # pragma: no cover - lightweight helper
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMForecaster:
    """
    Minimal forecaster wrapper.

    If PyTorch artifacts exist, the model is used; otherwise a heuristic
    fallback produces reasonable placeholder forecasts so pipelines keep
    running in lightweight environments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.model_path = Path(self.config.get("lstm_model_path", "models/lstm/model.pt"))
        self.scaler_path = Path(self.config.get("scaler_path", "models/lstm/scaler.pkl"))
        self.meta_path = Path(self.config.get("meta_path", "models/lstm/meta.json"))
        self.model: Any = None
        self.device = "cpu"
        self.input_dim = int(self.config.get("input_dim", 2))
        self.hidden_dim = int(self.config.get("hidden_dim", 8))
        self._load_artifacts()

    @classmethod
    def from_config(cls, config_path: str) -> "LSTMForecaster":
        cfg: Dict[str, Any] = {}
        path = Path(config_path)
        if path.exists():
            if yaml is not None and path.suffix.lower() in {".yml", ".yaml"}:
                cfg = yaml.safe_load(path.read_text()) or {}
            else:
                cfg = json.loads(path.read_text())
        cfg.setdefault("lstm_config_path", str(config_path))
        return cls(cfg)

    def _load_artifacts(self) -> None:
        if torch is None or nn is None:
            return
        if not self.model_path.exists():
            return
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            input_dim = checkpoint.get("input_dim", self.input_dim)
            hidden_dim = checkpoint.get("hidden_dim", self.hidden_dim)
            model_state = checkpoint.get("state_dict")
            model = _TinyLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
            if model_state:
                model.load_state_dict(model_state)
            model.eval()
            self.model = model.to(self.device)
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
        except Exception:
            self.model = None

    def _heuristic_forecast(self, features: np.ndarray) -> Dict[str, float]:
        """
        Simple statistical fallback when no model is available.
        """

        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        returns = np.diff(np.log(np.clip(x[:, 0], 1e-6, None)))
        if returns.size == 0:
            returns = np.array([0.0], dtype=float)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        prob_up = float(1.0 / (1.0 + np.exp(-mu / (sigma + 1e-6))))
        return {"next_return": mu, "vol_forecast": sigma, "prob_up": prob_up}

    def predict(self, features, horizon: int) -> Dict[str, float]:
        """
        features: np.ndarray or torch.Tensor [T, d]
        horizon: forecast horizon in steps
        """

        x_np = np.asarray(features, dtype=float)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)

        if self.model is None or torch is None:
            return self._heuristic_forecast(x_np)

        with torch.no_grad():
            tensor = torch.tensor(x_np[-horizon:], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(tensor)
            next_ret, vol, prob = logits.squeeze(0).cpu().numpy().tolist()
            prob_up = float(1.0 / (1.0 + np.exp(-prob)))
            return {
                "next_return": float(next_ret),
                "vol_forecast": float(abs(vol)),
                "prob_up": prob_up,
            }
