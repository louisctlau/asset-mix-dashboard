"""Live option chains — Yahoo Finance first, CBOE delayed as fallback.

2026-11-20: CBOE moved its delayed-quotes API to cdn-api.cboe.com and
put Cloudflare bot protection in front of it — every server-side fetch
(urllib, curl, even a real browser) now gets a 403 "Attention Required"
page, so the live CBOE feed is unusable from Streamlit Cloud. Yahoo
Finance option chains carry the same fields the app needs (bid/ask,
volume, open interest, IV) and work from here; the app computes its own
Black-Scholes greeks from IV anyway, so the Yahoo path fills the
cboe_delta / cboe_gamma columns with model values (r = 5%, q = 0) and
downstream code (volsurface skew, GEX) is untouched. CBOE stays as a
one-shot fallback in case it unblocks later.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date

import pandas as pd
import yfinance as yf

import greeks as _gk

TICKERS = ["NVDA", "AMD", "MU", "AAPL", "META", "GOOG"]  # archive universe;
# the app itself accepts any ticker with a listed option chain.
CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"

# Index options need CBOE's underscore-prefixed symbols; Yahoo quotes the
# cash indices with a ^ prefix. Plain tickers pass through unchanged.
INDEX_TICKERS = {
    "SPX": ("_SPX", "^SPX"),
    "RUT": ("_RUT", "^RUT"),
    "NDX": ("_NDX", "^NDX"),
    "VIX": ("_VIX", "^VIX"),
}

_RF = 0.05  # model-greeks risk-free rate (r barely moves delta/gamma)
_MAX_EXPIRIES = 20  # bounds the Yahoo fetch (~1s/expiry); term structure uses 12


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


def _model_greeks(spot: float, strike: float, expiry: str, iv: float,
                  kind: str) -> dict:
    """BS delta/gamma/theta/vega for one contract (r=5%, q=0)."""
    d = max((date.fromisoformat(expiry) - date.today()).days, 0.5) / 365.0
    try:
        g = _gk.greeks(spot, strike, d, _RF, 0.0, max(iv, 1e-4), kind)
        return {"cboe_delta": g["delta"], "cboe_gamma": g["gamma"],
                "cboe_theta": g["theta"], "cboe_vega": g["vega"]}
    except Exception:  # noqa: BLE001 - leave greeks blank, keep the row
        return {"cboe_delta": None, "cboe_gamma": None,
                "cboe_theta": None, "cboe_vega": None}


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop alternate-root duplicates (B1SPY next to SPY etc.) keeping the
    most liquid contract per (expiry, strike, type)."""
    if df.empty:
        return df
    return (df.assign(_vol=df["volume"].fillna(0),
                      _oi=df["open_interest"].fillna(0))
              .sort_values(["_vol", "_oi"], ascending=False)
              .drop_duplicates(subset=["expiry", "strike", "option_type"])
              .drop(columns=["_vol", "_oi"]).reset_index(drop=True))


def _fetch_yahoo(yahoo_sym: str,
                 max_expiries: int = _MAX_EXPIRIES) -> tuple[pd.DataFrame, float]:
    """Full chain from Yahoo Finance option chains (one request/expiry).

    Retries 3x with backoff: Yahoo rate-limits Streamlit Cloud's shared
    egress IPs intermittently (t.options comes back empty), and a retry
    a few seconds later usually succeeds.
    """
    last = None
    for attempt in range(3):
        try:
            return _fetch_yahoo_once(yahoo_sym, max_expiries)
        except Exception as e:  # noqa: BLE001 - retry, then report
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Yahoo failed for {yahoo_sym} after 3 tries: {last}")


def _fetch_yahoo_once(yahoo_sym: str,
                      max_expiries: int = _MAX_EXPIRIES) -> tuple[pd.DataFrame, float]:
    t = yf.Ticker(yahoo_sym)
    expiries = list(t.options or [])
    if not expiries:
        raise RuntimeError(f"Yahoo returned no expiries for {yahoo_sym}")
    hist = t.history(period="1d")
    if hist is not None and not hist.empty:
        spot = float(hist["Close"].iloc[-1])
    else:
        spot = float(t.fast_info.get("last_price") or 0) or None
    if not spot:
        raise RuntimeError(f"Yahoo returned no spot for {yahoo_sym}")
    rows = []
    for exp in expiries[:max_expiries]:
        try:
            chain = t.option_chain(exp)
        except Exception:  # noqa: BLE001 - skip an expiry that fails
            continue
        for kind, frame in (("call", chain.calls), ("put", chain.puts)):
            for _, row in frame.iterrows():
                iv = row.get("impliedVolatility")
                if iv is None or not (0 < float(iv) < 5):
                    continue
                strike = float(row["strike"])
                rows.append({
                    "expiry": exp,
                    "option_type": kind,
                    "strike": strike,
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "volume": row.get("volume"),
                    "open_interest": row.get("openInterest"),
                    "iv": float(iv),
                    **_model_greeks(spot, strike, exp, float(iv), kind),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no usable contracts for {yahoo_sym}")
    df["expiry"] = pd.to_datetime(df["expiry"])
    for c in ("bid", "ask", "iv", "cboe_delta", "cboe_gamma",
              "cboe_theta", "cboe_vega"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return _dedupe(df), spot


def _fetch_cboe(cboe_sym: str) -> tuple[pd.DataFrame, float | None]:
    """Full chain from CBOE delayed quotes (one quick try — used as a
    fallback now that Cloudflare blocks it from servers)."""
    data = _get(CBOE_URL.format(cboe_sym), tries=1)["data"]
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
        df = _dedupe(df)
    return df, (float(und) if und else None)


def fetch_chain(ticker: str) -> tuple[pd.DataFrame, float | None, str]:
    """Fetch the full chain for one ticker, Yahoo first, CBOE fallback.

    Returns (chain DataFrame, underlying price or None, source name).
    Raises RuntimeError if both sources fail.
    """
    cboe_sym, yahoo_sym = resolve_ticker(ticker)
    errors = []
    try:
        df, spot = _fetch_yahoo(yahoo_sym)
        if not df.empty and spot:
            return df, spot, "Yahoo Finance"
        errors.append("Yahoo: empty chain")
    except Exception as e:  # noqa: BLE001 - fall through to CBOE
        errors.append(f"Yahoo: {e}")
    try:
        df, spot = _fetch_cboe(cboe_sym)
        if not df.empty and spot:
            return df, spot, "CBOE"
        errors.append("CBOE: empty chain")
    except Exception as e:  # noqa: BLE001 - report both failures
        errors.append(f"CBOE: {e}")
    raise RuntimeError(f"No chain data for {ticker} ({'; '.join(errors)})")
