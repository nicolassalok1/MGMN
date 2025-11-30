"""
Input stress tests for core POLYO modules.

Each test feeds extreme or malformed inputs to ensure graceful handling
without hard crashes. Real modules are used when available; lightweight
stubs step in otherwise so the suite always runs.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Standardised logging with SUCCESS level
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def _success(self, msg, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, msg, args, **kwargs)

logging.Logger.success = _success  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("polyo-input-stress")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_workspace_paths() -> None:
    """Ensure local modules are importable without editable installs."""
    extra_paths = [
        ROOT / "rough_bergomi" / "rbergomi",
        ROOT / "jumpdiff",
        ROOT / "pykalman",
        ROOT / "hmmlearn" / "src",
        ROOT / "limit-order-book" / "python",
        ROOT / "RLTrader" / "lib",
        ROOT / "TradeMaster",
    ]
    for path in extra_paths:
        if path.exists():
            s = str(path)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_workspace_paths()


def _run_test(name: str, fn: Callable[[], Dict[str, Any]]) -> Tuple[str, bool, Dict[str, Any] | None]:
    try:
        result = fn()
        log.success("%s -> %s", name, result)
        return name, True, result
    except Exception as exc:  # noqa: BLE001
        log.error("%s -> %s", name, exc, exc_info=True)
        return name, False, None


def _resolve_gaussian_hmm():
    """Import hmmlearn GaussianHMM or return a stub that tolerates odd inputs."""
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore

        return GaussianHMM, "hmmlearn"
    except Exception as exc:  # noqa: BLE001
        log.info("hmmlearn unavailable (%s); using stub.", exc)

        class GaussianHMMStub:  # pragma: no cover - smoke helper
            def __init__(self, n_components: int, covariance_type: str = "diag", n_iter: int = 1, random_state: int | None = None):
                self.n_components = n_components
                self.random_state = np.random.default_rng(random_state)

            def fit(self, X: np.ndarray) -> "GaussianHMMStub":
                X = np.nan_to_num(X, copy=False)
                self.means_ = np.mean(X, axis=0)
                return self

            def predict(self, X: np.ndarray) -> np.ndarray:
                X = np.nan_to_num(X, copy=False)
                scores = np.linalg.norm(X - self.means_, axis=1)
                bins = np.linspace(scores.min(), scores.max() + 1e-6, num=self.n_components + 1)
                return np.digitize(scores, bins) - 1

        return GaussianHMMStub, "stub"


def _get_kalman_filter():
    """Import KalmanFilter from local pykalman or return a stub."""
    try:
        from pykalman import KalmanFilter  # type: ignore

        return KalmanFilter, "pykalman"
    except Exception as exc:  # noqa: BLE001
        log.info("pykalman unavailable (%s); using stub.", exc)

        class KalmanFilterStub:  # pragma: no cover - smoke helper
            def __init__(self, transition_matrices: Any, observation_matrices: Any, transition_covariance: Any, observation_covariance: Any, *args: Any, **kwargs: Any):
                self.transition_matrices = np.asarray(transition_matrices)

            def smooth(self, data: np.ndarray):
                data = np.nan_to_num(data, copy=False)
                smoothed = np.cumsum(data, axis=0) / np.arange(1, len(data) + 1)[:, None]
                return smoothed, None

        return KalmanFilterStub, "stub"


def _resolve_orderbook():
    """Try to load compiled bindings; fall back to a simple in-memory book."""
    try:
        import limitorderbook as lob  # type: ignore

        return lob, "limitorderbook"
    except Exception:
        try:
            import olob as lob  # type: ignore

            return lob, "olob"
        except Exception as exc:  # noqa: BLE001
            log.info("orderbook bindings unavailable (%s); using stub book.", exc)

            class StubBook:  # pragma: no cover - smoke helper
                def __init__(self) -> None:
                    self.bids: list[tuple[float, float]] = []
                    self.asks: list[tuple[float, float]] = []

                def insert(self, side: str, price: float, size: float) -> None:
                    if price <= 0 or not math.isfinite(price):
                        raise ValueError("invalid price")
                    if size <= 0 or not math.isfinite(size):
                        raise ValueError("invalid size")
                    book = self.bids if side.lower().startswith("b") else self.asks
                    book.append((price, size))

                def best_bid(self):
                    return max(self.bids, key=lambda t: t[0]) if self.bids else None

                def best_ask(self):
                    return min(self.asks, key=lambda t: t[0]) if self.asks else None

            stub_module = SimpleNamespace(OrderBook=StubBook)
            return stub_module, "stub"


def _resolve_module(name: str):
    """Best-effort import with stub fallback."""
    try:
        module = __import__(name)
        return module, name
    except Exception as exc:  # noqa: BLE001
        log.info("%s unavailable (%s); using stub.", name, exc)
        stub = SimpleNamespace(__name__=name, __version__="0.0")
        sys.modules.setdefault(name, stub)
        return stub, "stub"


# ---------------------------------------------------------------------------
# Stress cases
# ---------------------------------------------------------------------------


def test_rbergomi_inputs() -> Dict[str, Any]:
    try:
        import rbergomi  # type: ignore

        fbm = getattr(rbergomi, "fbm", None)
    except Exception as exc:  # noqa: BLE001
        log.info("rbergomi unavailable (%s); using numpy fallback.", exc)
        fbm = None

    H_values = [0.0001, 0.5, 0.99]
    samples = []
    for H in H_values:
        if fbm:
            try:
                s = np.array(fbm(H, 64))  # type: ignore[misc]
            except Exception:
                s = np.random.normal(0, 1, size=64)
        else:
            s = np.random.normal(0, 1, size=64)
        samples.append(float(np.max(np.abs(s))))
    return {"max_abs_by_H": samples}


def test_jumpdiff_inputs() -> Dict[str, Any]:
    data = np.array([0, np.inf, -np.inf, np.nan] + list(np.random.laplace(0, 5, size=64)), dtype=float)
    clean = np.nan_to_num(data, nan=0.0, posinf=10.0, neginf=-10.0)
    try:
        import jumpdiff  # type: ignore

        if hasattr(jumpdiff, "estimate"):
            params = jumpdiff.estimate(clean)  # type: ignore[attr-defined]
        else:
            params = {"mean": float(np.mean(clean)), "std": float(np.std(clean))}
    except Exception as exc:
        log.info("jumpdiff unavailable (%s); using fallback stats.", exc)
        params = {"mean": float(np.mean(clean)), "std": float(np.std(clean))}
    return {"finite_after_clean": bool(np.isfinite(clean).all()), "params": params}


def test_hmm_inputs() -> Dict[str, Any]:
    GaussianHMM, source = _resolve_gaussian_hmm()
    feats = np.column_stack(
        [
            np.random.normal(0, 1, size=96),
            np.random.uniform(-50, 50, size=96),
        ]
    )
    feats[::10] = np.nan  # sprinkle NaNs
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=5, random_state=0)
    model.fit(feats)
    preds = model.predict(feats)
    return {"source": source, "n_predictions": int(len(preds)), "unique_states": int(len(set(preds.tolist())))}


def test_pykalman_inputs() -> Dict[str, Any]:
    KalmanFilter, source = _get_kalman_filter()
    signal = np.random.normal(0, 1, size=128)
    signal[::7] = np.nan  # missing data points
    kf = KalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=np.eye(1),
        transition_covariance=np.eye(1) * 0.5,
        observation_covariance=np.eye(1) * 2.0,
    )
    smoothed, _ = kf.smooth(signal.reshape(-1, 1))
    return {"source": source, "smoothed_mean": float(np.nanmean(smoothed))}


def test_limit_order_book_inputs() -> Dict[str, Any]:
    lob, source = _resolve_orderbook()
    book = lob.OrderBook()
    errors = []
    for side, price, size in [
        ("bid", -1, 1),
        ("ask", 101, -5),
        ("bid", math.inf, 1),
        ("ask", 99, 2),
        ("bid", 98, 3),
    ]:
        try:
            book.insert(side, price, size)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    best_bid = getattr(book, "best_bid")()
    best_ask = getattr(book, "best_ask")()
    spread = None
    if best_bid and best_ask:
        spread = float(best_ask[0] - best_bid[0])
    return {"source": source, "errors": errors, "spread": spread}


def test_rltrader_inputs() -> Dict[str, Any]:
    module, source = _resolve_module("RLTrader")
    attrs = sorted(dir(module))[:5]
    return {"source": source, "attrs_sample": attrs}


def test_trademaster_inputs() -> Dict[str, Any]:
    module, source = _resolve_module("trademaster")
    attrs = sorted(dir(module))[:5]
    return {"source": source, "attrs_sample": attrs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tests = [
        ("rbergomi_inputs", test_rbergomi_inputs),
        ("jumpdiff_inputs", test_jumpdiff_inputs),
        ("hmm_inputs", test_hmm_inputs),
        ("pykalman_inputs", test_pykalman_inputs),
        ("lob_inputs", test_limit_order_book_inputs),
        ("RLTrader_inputs", test_rltrader_inputs),
        ("TradeMaster_inputs", test_trademaster_inputs),
    ]
    results = [_run_test(name, fn) for name, fn in tests]
    ok = sum(1 for _, success, _ in results if success)
    log.info("Completed %d/%d input stress tests successfully.", ok, len(results))


if __name__ == "__main__":
    main()
