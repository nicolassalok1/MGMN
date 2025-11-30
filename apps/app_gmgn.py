from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional local imports (POLYO submodules) if ever needed downstream
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rough_bergomi"))
sys.path.insert(0, str(ROOT / "jumpdiff"))
sys.path.insert(0, str(ROOT / "pykalman"))
sys.path.insert(0, str(ROOT / "hmmlearn" / "src"))
sys.path.insert(0, str(ROOT / "limit-order-book" / "python"))
sys.path.insert(0, str(ROOT / "RLTrader"))
sys.path.insert(0, str(ROOT / "TradeMaster"))

BASE_URL = "https://gmgn.ai/api/v1/solana"


def gmgn_request(endpoint: str, params: Dict[str, Any] | None = None, api_key: Optional[str] = None) -> Tuple[int, Any]:
    params = params or {}
    headers = {}
    if api_key:
        headers["gmgn-api-key"] = api_key
    url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        code = resp.status_code
        data = resp.json() if resp.content else {}
        if code != 200:
            st.warning(f"GMGN API returned {code}: {data}")
        return code, data
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
        return 0, None


@st.cache_data(ttl=60)
def fetch_new_pairs(api_key: Optional[str]) -> Any:
    return gmgn_request("/pairs/new", api_key=api_key)


@st.cache_data(ttl=60)
def fetch_token_info(token_addr: str, api_key: Optional[str]) -> Any:
    return gmgn_request(f"/token/{token_addr}", api_key=api_key)


@st.cache_data(ttl=30)
def fetch_trades(token_addr: str, api_key: Optional[str]) -> Any:
    return gmgn_request(f"/token/{token_addr}/trades", api_key=api_key)


@st.cache_data(ttl=120)
def fetch_market_list(list_type: str, api_key: Optional[str]) -> Any:
    endpoint_map = {
        "gainers": "/market/gainers",
        "losers": "/market/losers",
        "volume": "/market/volume",
        "trending": "/market/trending",
    }
    ep = endpoint_map.get(list_type, "/market/trending")
    return gmgn_request(ep, api_key=api_key)


def render_new_pairs(api_key: Optional[str]):
    st.subheader("New Pairs Scanner")
    if st.button("Refresh New Pairs"):
        code, data = fetch_new_pairs(api_key)
        if code == 200 and data:
            pairs = data.get("data") or data
            df = pd.DataFrame(pairs)
            display_cols = [c for c in df.columns if c.lower() in {"token", "token_name", "token_address", "pool_address", "created_at", "liquidity", "fdv", "holders"}]
            if not display_cols:
                display_cols = df.columns
            st.dataframe(df[display_cols])
            if len(df) > 0:
                token_col = "token_address" if "token_address" in df.columns else df.columns[0]
                selection = st.selectbox("Select a token to inspect", df[token_col])
                details = df[df[token_col] == selection].to_dict(orient="records")[0]
                st.json(details)
        else:
            st.warning("No data returned.")
    else:
        st.info("Click 'Refresh New Pairs' to query GMGN.")


def render_token_info(api_key: Optional[str]):
    st.subheader("Token Info")
    token_addr = st.text_input("Token address (Solana)")
    if st.button("Fetch Token Info") and token_addr:
        code, data = fetch_token_info(token_addr, api_key)
        if code == 200 and data:
            info = data.get("data") or data
            price = info.get("price") or info.get("price_usd")
            liq = info.get("liquidity")
            vol = info.get("volume_24h") or info.get("volume")
            holders = info.get("holders")
            cols = st.columns(4)
            cols[0].metric("Price", f"{price:.6f}" if price else "N/A")
            cols[1].metric("Liquidity", f"{liq:,.0f}" if liq else "N/A")
            cols[2].metric("24h Volume", f"{vol:,.0f}" if vol else "N/A")
            cols[3].metric("Holders", f"{holders:,}" if holders else "N/A")

            base = float(price) if price else 1.0
            series = base * (1 + np.random.normal(0, 0.01, size=50)).cumprod()
            st.line_chart(series)
            st.json(info)
        else:
            st.warning("No data returned or request failed.")
    elif token_addr == "":
        st.info("Enter a token address and click Fetch.")


def render_trades(api_key: Optional[str]):
    st.subheader("Live Trades Feed")
    token_addr = st.text_input("Token address for trades (Solana)")
    if st.button("Refresh Trades") and token_addr:
        code, data = fetch_trades(token_addr, api_key)
        if code == 200 and data:
            trades = data.get("data") or data
            df = pd.DataFrame(trades)
            if not df.empty:
                keep = [c for c in df.columns if c.lower() in {"side", "amount", "usd_amount", "price", "ts", "timestamp"}]
                st.dataframe(df[keep] if keep else df)
                if "usd_amount" in df.columns:
                    st.bar_chart(df["usd_amount"])
            else:
                st.warning("No trades available.")
        else:
            st.warning("Request failed or empty response.")
    else:
        st.info("Enter token address then click Refresh.")


def render_market_dashboard(api_key: Optional[str]):
    st.subheader("Market Dashboard")
    cols = st.columns(4)
    categories = ["gainers", "losers", "volume", "trending"]
    data_map = {}
    for cat, col in zip(categories, cols):
        code, data = fetch_market_list(cat, api_key)
        entries = (data.get("data") if data else None) if code == 200 else None
        if entries:
            df = pd.DataFrame(entries)
            display_cols = [c for c in df.columns if c.lower() in {"token", "token_name", "token_address", "price", "change", "volume"}]
            if display_cols:
                df_small = df[display_cols].head(5)
            else:
                df_small = df.head(5)
            col.write(cat.capitalize())
            col.dataframe(df_small)
            data_map[cat] = df
        else:
            col.write(f"{cat.capitalize()}: no data")
    st.divider()
    if st.button("Show Top Gainers Chart") and "gainers" in data_map:
        df = data_map["gainers"]
        metric_col = "volume" if "volume" in df.columns else df.columns[1]
        st.bar_chart(df.head(10).set_index(df.columns[0])[metric_col])


def simulate_trading(token: str, balance: float) -> Dict[str, Any]:
    prices = 1 + np.cumsum(np.random.normal(0, 0.02, size=10))
    position = 0.0
    cash = balance
    history = []
    for i, p in enumerate(prices):
        action = np.random.choice(["buy", "sell", "hold"])
        if action == "buy" and cash > 0:
            qty = cash / p * 0.5
            cash -= qty * p
            position += qty
        elif action == "sell" and position > 0:
            qty = position * 0.5
            cash += qty * p
            position -= qty
        value = cash + position * p
        history.append({"step": i, "price": p, "action": action, "cash": cash, "position": position, "equity": value})
    return {"history": history, "final_equity": history[-1]["equity"], "token": token}


def render_rl_demo(api_key: Optional[str]):
    st.subheader("Simulation / Paper Trading")
    token = st.text_input("Token (for context only)", value="SOL")
    balance = st.number_input("Starting balance (USD)", min_value=10.0, value=1000.0, step=50.0)
    if st.button("Simulate 10 steps"):
        result = simulate_trading(token, balance)
        df = pd.DataFrame(result["history"])
        st.dataframe(df)
        st.line_chart(df.set_index("step")[["equity", "price"]])
        st.success(f"Final equity: {result['final_equity']:.2f} USD")


# --------------------------------------------------------------------
# Layout
st.set_page_config(page_title="GMGN Solana Trading UI", layout="wide")
st.title("GMGN Solana Trading UI (Simulated)")
api_key_input = st.sidebar.text_input("GMGN API Key", type="password")

tabs = st.tabs(
    [
        "New Pairs Scanner",
        "Token Info",
        "Live Trades Feed",
        "Market Dashboard",
        "Simulation / Paper Trading",
    ]
)

with tabs[0]:
    render_new_pairs(api_key_input)

with tabs[1]:
    render_token_info(api_key_input)

with tabs[2]:
    render_trades(api_key_input)

with tabs[3]:
    render_market_dashboard(api_key_input)

with tabs[4]:
    render_rl_demo(api_key_input)
