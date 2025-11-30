from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None

from polyo.lstm_forecaster.model import _TinyLSTM


def _load_config(path: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if not path.exists():
        return cfg
    if yaml is not None and path.suffix.lower() in {".yml", ".yaml"}:
        cfg = yaml.safe_load(path.read_text()) or {}
    else:
        cfg = json.loads(path.read_text())
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny LSTM forecaster (mock).")
    parser.add_argument("--config", type=str, default="config/lstm.yaml", help="Path to LSTM YAML config.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    model_dir = Path(cfg.get("model_dir", "models/lstm"))
    model_dir.mkdir(parents=True, exist_ok=True)

    input_dim = int(cfg.get("input_dim", 2))
    hidden_dim = int(cfg.get("hidden_dim", 8))
    epochs = int(cfg.get("epochs", 2))
    horizon = int(cfg.get("horizon", 8))

    # Synthetic training data (kept tiny to stay light)
    rng = np.random.default_rng(42)
    series = rng.normal(0, 0.02, size=(128, input_dim))
    targets = series.sum(axis=1)

    if torch is not None and nn is not None:
        model = _TinyLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
        loss_fn = nn.MSELoss()
        x_tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(targets, dtype=torch.float32)
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(x_tensor).squeeze(0)[:, 0]
            loss = loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step()
        torch.save(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "state_dict": model.state_dict(),
            },
            model_dir / "model.pt",
        )

    # Minimal scaler + meta files so downstream calls find expected artifacts
    (model_dir / "scaler.pkl").write_bytes(b"placeholder-scaler")
    meta = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "horizon": horizon,
        "config_path": str(cfg_path),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved LSTM mock artifacts to {model_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
