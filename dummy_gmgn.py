from __future__ import annotations

import numpy as np
import pandas as pd


def _seed_from_string(text: str, offset: int = 0) -> int:
    total = 0
    for ch in text:
        total = (total * 131 + ord(ch)) % 1_000_000_007
    return (total + offset + 42) % (2**32)


def get_dummy_new_pairs() -> dict:
    rng = np.random.default_rng(42)
    base_time = pd.Timestamp("2024-04-01T00:00:00Z")
    symbols = ["BONK", "SAGA", "WIF", "MEOW", "HADES", "PYTH", "BORK"]
    pairs = []
    for idx, sym in enumerate(symbols):
        created = base_time + pd.Timedelta(minutes=idx * 7)
        liquidity = float(rng.uniform(80_000, 250_000))
        volume = float(rng.uniform(150_000, 650_000))
        price = float(rng.uniform(0.0005, 2.5))
        entry = {
            "tokenSymbol": sym,
            "tokenName": f"{sym} Token",
            "mint": f"{sym.lower():<4}MintAddr{idx:02d}XYZ1234567890",
            "pairAddress": f"Pair{sym}Addr{idx:02d}ABCDEFGH987654321",
            "created_at": created.isoformat(),
            "liquidity": round(liquidity, 2),
            "volume_24h": round(volume, 2),
            "price": round(price, 6),
            "fdv": round(liquidity * 8.5, 2),
        }
        pairs.append(entry)
    return {"code": 0, "msg": "success", "data": pairs}


def get_dummy_token_info(token_address: str) -> dict:
    seed = _seed_from_string(token_address, offset=7)
    rng = np.random.default_rng(seed)
    price = float(rng.uniform(0.01, 3.0))
    liquidity = float(rng.uniform(50_000, 400_000))
    volume = float(rng.uniform(80_000, 900_000))
    holders = int(rng.integers(1_000, 25_000))
    fdv = price * rng.uniform(10_000_000, 50_000_000) * 1e-2
    swaps = int(rng.integers(500, 4_000))
    risks = ["volatile", "new_listing"] if price < 0.1 else ["liquidity_ok"]
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "token": token_address,
            "symbol": token_address[:4].upper(),
            "price": round(price, 6),
            "priceChange": round(rng.normal(0.04, 0.02), 4),
            "liquidity": round(liquidity, 2),
            "volume": round(volume, 2),
            "fdv": round(fdv, 2),
            "holders": holders,
            "risks": risks,
            "swaps_24h": swaps,
        },
    }


def get_dummy_trades(token_address: str) -> list:
    seed = _seed_from_string(token_address, offset=11)
    rng = np.random.default_rng(seed)
    n_trades = int(rng.integers(15, 31))
    base_ts = pd.Timestamp("2024-04-01T12:00:00Z").timestamp()
    trades = []
    for i in range(n_trades):
        side = rng.choice(["buy", "sell"])
        amount_in = float(rng.uniform(50, 5000))
        amount_out = float(amount_in * rng.uniform(0.9, 1.1))
        usd_value = float(amount_in * rng.uniform(0.9, 1.2))
        ts = int(base_ts + i * rng.integers(30, 120))
        wallet = f"Wallet{rng.integers(1000, 9999)}{i:02d}XYZ"
        trades.append(
            {
                "side": side,
                "amountIn": round(amount_in, 4),
                "amountOut": round(amount_out, 4),
                "usdValue": round(usd_value, 4),
                "timestamp": ts,
                "wallet": wallet,
            }
        )
    return trades


def get_dummy_price_series(n_points: int = 200) -> dict:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq="T")
    trend = 0.0008
    noise = rng.normal(0, 0.01, size=n_points)
    increments = trend + noise
    prices = 1.0 * np.exp(np.cumsum(increments))
    rets = np.zeros(n_points)
    rets[1:] = np.diff(np.log(prices))
    return {
        "timestamps": [ts.isoformat() for ts in timestamps],
        "prices": prices.round(6).tolist(),
        "returns": rets.round(6).tolist(),
    }


def get_dummy_rough_vol_path(n_steps: int = 128) -> dict:
    rng = np.random.default_rng(42)
    H = 0.18
    eta = 0.7
    noise = rng.normal(0, 1, size=n_steps)
    weights = (np.arange(1, 9) ** (H - 0.5)).astype(float)
    conv = np.convolve(noise, weights, mode="same")
    vol = 0.2 + 0.05 * conv
    vol = np.maximum(0.01, vol)
    return {"vol_path": vol.round(6).tolist(), "params": {"H": H, "eta": eta}}


def get_dummy_jumpdiff_returns(n_points: int = 200) -> list:
    rng = np.random.default_rng(42)
    base = rng.normal(0, 0.01, size=n_points)
    jump_mask = rng.random(n_points) < 0.05
    jumps = rng.normal(0, 0.08, size=n_points) * jump_mask
    returns = base + jumps
    return returns.round(6).tolist()


def get_dummy_kalman_state_series(n_points: int = 200) -> dict:
    rng = np.random.default_rng(42)
    t = np.arange(n_points)
    latent = 0.1 * np.sin(np.linspace(0, 4 * np.pi, n_points)) + 0.0005 * t
    noise = rng.normal(0, 0.05, size=n_points)
    observed = latent + noise
    smoothed = pd.Series(observed).ewm(span=5, adjust=False).mean().values
    return {
        "observed": observed.round(6).tolist(),
        "smoothed": smoothed.round(6).tolist(),
        "latent": latent.round(6).tolist(),
    }


def get_dummy_regimes(n_points: int = 200, n_regimes: int = 3) -> dict:
    rng = np.random.default_rng(42)
    ret = rng.normal(0, 0.01, size=n_points) + 0.0002 * np.arange(n_points)
    vol = np.abs(rng.normal(0.2, 0.02, size=n_points))
    features = np.column_stack([ret, vol])
    bins = np.quantile(ret, np.linspace(0, 1, n_regimes + 1)[1:-1])
    regimes = np.digitize(ret, bins).astype(int)
    return {"features": features.round(6).tolist(), "regimes": regimes.tolist()}


def get_dummy_rl_episode(n_steps: int = 15) -> dict:
    rng = np.random.default_rng(42)
    trend = 0.002
    noise = rng.normal(0, 0.01, size=n_steps + 1)
    increments = trend + noise
    prices = 1.0 * np.exp(np.cumsum(increments))
    actions = rng.integers(0, 3, size=n_steps)
    positions = np.where(actions == 1, 1.0, np.where(actions == 2, -1.0, 0.0))
    deltas = prices[1:] - prices[:-1]
    rewards = positions * deltas
    regimes = np.mod(np.arange(n_steps), 3)
    cum_pnl = np.cumsum(rewards)
    return {
        "t": list(range(n_steps)),
        "price": prices[1:].round(6).tolist(),
        "regime": regimes.tolist(),
        "action": actions.tolist(),
        "reward": rewards.round(6).tolist(),
        "cum_pnl": cum_pnl.round(6).tolist(),
    }
