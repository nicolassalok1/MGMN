from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyo.config import PolyoConfig  # noqa: E402


def test_streamlit_like_config_toggle():
    cfg = PolyoConfig()
    assert not cfg.use_lstm

    overrides = {
        "use_lstm": True,
        "lstm_horizon": 12,
        "lstm_features": ["returns", "price"],
        "lstm_model_path": "models/lstm/model.pt",
    }
    cfg.update_from_dict(overrides)
    cfg.merge_features(["volume"])

    as_dict = cfg.to_dict()
    assert as_dict["use_lstm"] is True
    assert as_dict["lstm_horizon"] == 12
    assert "volume" in as_dict["lstm_features"]
    assert as_dict["lstm_model_path"].endswith("model.pt")
