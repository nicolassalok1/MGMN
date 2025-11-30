from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_signal_pipeline import (  # noqa: E402
    OpenAISentimentClient,
    ReliabilityLearner,
    SignalPrediction,
    TelegramMessage,
    TelegramScraperAdapter,
)


def test_sentiment_heuristic_extracts_tokens():
    msg = TelegramMessage(channel="alpha", message_id=1, text="Buy SOL and WIF now", timestamp=None)
    client = OpenAISentimentClient(api_key=None)
    signals = client.analyze(msg)
    tokens = {s.token for s in signals}
    assert {"SOL", "WIF"} <= tokens
    assert all(s.source == "alpha" for s in signals)


def test_aggregate_respects_source_weights():
    learner = ReliabilityLearner(initial_weights={"alpha": 2.0, "beta": 0.5})
    signals = [
        SignalPrediction(
            token="SOL",
            stance="buy",
            confidence=0.9,
            sentiment_score=1.0,
            source="chanA",
            source_type="alpha",
            reason="",
            message_excerpt="",
        ),
        SignalPrediction(
            token="SOL",
            stance="sell",
            confidence=0.5,
            sentiment_score=-0.5,
            source="chanB",
            source_type="beta",
            reason="",
            message_excerpt="",
        ),
    ]
    aggregated = learner.aggregate(signals)
    assert aggregated[0]["token"] == "SOL"
    assert aggregated[0]["score"] == pytest.approx(1.75)  # 1.0*2.0 + (-0.5*0.5)


def test_fetcher_loads_json_exports():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        channel = "testchannel"
        chan_dir = root / channel
        chan_dir.mkdir(parents=True, exist_ok=True)
        json_path = chan_dir / f"{channel}.json"
        sample = [
            {"message_id": 10, "message": "Bullish on SOL", "date": "2024-01-01 00:00:00"},
        ]
        json_path.write_text(json.dumps(sample), encoding="utf-8")

        fetcher = TelegramScraperAdapter(export_root=root, api_id=None, api_hash=None)
        messages = fetcher.fetch_messages([channel], limit=5)

        assert len(messages) == 1
        assert messages[0].text == "Bullish on SOL"
        assert messages[0].channel == channel
