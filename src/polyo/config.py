from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class PolyoConfig:
    """
    Centralised configuration used by the toy decision/RL pipeline.

    Only a handful of fields are needed for the LSTM integration; everything
    else falls back to defaults so existing flows keep working when LSTM is
    disabled.
    """

    use_lstm: bool = False
    lstm_horizon: int = 8
    lstm_features: List[str] = field(default_factory=lambda: ["returns"])
    lstm_model_path: str = "models/lstm/model.pt"
    lstm_config_path: str = "config/lstm.yaml"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_lstm": self.use_lstm,
            "lstm_horizon": int(self.lstm_horizon),
            "lstm_features": list(self.lstm_features),
            "lstm_model_path": self.lstm_model_path,
            "lstm_config_path": self.lstm_config_path,
        }

    def update_from_dict(self, overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PolyoConfig":
        cfg = {}
        p = Path(path)
        if p.exists() and yaml is not None:
            loaded = yaml.safe_load(p.read_text()) or {}
            cfg.update(loaded if isinstance(loaded, dict) else {})
        elif p.exists():
            loaded = json.loads(p.read_text())
            cfg.update(loaded if isinstance(loaded, dict) else {})
        return cls(**cfg)

    def merge_features(self, extra: Iterable[str]) -> None:
        """Update lstm_features with additional entries while keeping order."""
        current = list(self.lstm_features)
        for feat in extra:
            if feat not in current:
                current.append(feat)
        self.lstm_features = current

    @property
    def lstm_enabled(self) -> bool:
        return bool(self.use_lstm)
