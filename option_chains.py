"""Live CBOE delayed option chains (keyless, ~15 min delayed).

Same endpoint as the daily archive fetcher (options-chain-archive), but
returns the chain for one ticker on demand for the app. No snapshot
columns — the app caches with st.cache_data instead.
"""
from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

TICKERS = ["NVDA", "AMD", "MU", "AAPL", "META", "GOOG"]  # archive universe;
# the app itself accepts any ticker with a CBOE delayed chain.
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"

# Index options need CBOE's underscore-prefixed symbols; Yahoo quotes the
# cash indices with a ^ prefix. Plain tickers pass through unchanged.
INDEX_TICKERS = {
    "SPX": ("_SPX", "^SPX"),
    "RUT": ("_RUT", "^RUT"),
    "NDX": ("_NDX", "^NDX"),
    "VIX": ("_VIX", "^VIX"),
}


def resolve_ticker(ticker: str) -> tuple[str, str]:
    """Map a user-entered ticker to (CBOE symbol, Yahoo symbol)."""
    t = ticker.strip().upper()
    return INDEX_TICKERS.get(t, (t, t))


def _get(url: str, tries: int = 3) -> dict:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception as e:  # noqa: BLE001 - retry then report
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"CBOE fetch failed after {tries} tries: {last}")


def _parse_occ(sym: str) -> tuple[str, str, float]:
    """OCC symbol like NVDA260928C00100000 -> (expiry, type, strike)."""
    core = sym.strip()
    expiry = f"20{core[-15:-13]}-{core[-13:-11]}-{core[-11:-9]}"
    otype = "call" if core[-9] == "C" else "put"
    strike = int(core[-8:]) / 1000.0
    return expiry, otype, strike


def fetch_chain(ticker: str) -> tuple[pd.DataFrame, float | None]:
    """Fetch the full delayed chain for one ticker.

    Returns (chain DataFrame, underlying price or None). Raises
    RuntimeError if the fetch fails.
    """
    data = _get(CBOE_URL.format(ticker.upper()))["data"]
    und = data.get("current_price") or data.get("close")
    rows = []
    for o in data.get("options", []):
        try:
            expiry, otype, strike = _parse_occ(o["option"])
        except Exception:  # noqa: BLE001 - skip malformed symbols
            continue
        rows.append({
            "expiry": expiry,
            "option_type": otype,
            "strike": strike,
            "bid": o.get("bid"),
            "ask": o.get("ask"),
            "volume": o.get("volume"),
            "open_interest": o.get("open_interest"),
            "iv": o.get("iv"),
            "cboe_delta": o.get("delta"),
            "cboe_gamma": o.get("gamma"),
            "cboe_theta": o.get("theta"),
            "cboe_vega": o.get("vega"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["expiry"] = pd.to_datetime(df["expiry"])
        for c in ("bid", "ask", "iv", "cboe_delta", "cboe_gamma",
                  "cboe_theta", "cboe_vega"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, (float(und) if und else None)
