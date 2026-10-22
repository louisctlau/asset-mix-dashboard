"""Helpers for the Options Analytics history panels (IV percentile, GEX history).

Reads data/options_history.csv (built by scripts/build_options_history.py).
Everything degrades gracefully: missing file or too few points -> None, and
the app hides the panel silently.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HIST_CSV = Path(__file__).resolve().parent / "data" / "options_history.csv"
MIN_POINTS = 5


@lru_cache(maxsize=1)
def load_history() -> pd.DataFrame | None:
    """Full options history, or None if the CSV is absent/unreadable."""
    try:
        df = pd.read_csv(HIST_CSV, parse_dates=["date"])
    except Exception:
        return None
    if df.empty:
        return None
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def ticker_history(ticker: str) -> pd.DataFrame | None:
    """History rows for one ticker (upper-cased), or None if < MIN_POINTS."""
    df = load_history()
    if df is None:
        return None
    sub = df[df["ticker"] == ticker.upper()].copy()
    if len(sub) < MIN_POINTS:
        return None
    return sub.reset_index(drop=True)


def iv_percentile_row(ticker: str,
                      current_iv: float | None = None) -> dict | None:
    """Current ATM IV + its percentile rank vs trailing history.

    Percentile = fraction of history at-or-below current, 0-100. When the
    caller passes the LIVE ATM IV (from the freshly fetched chain) it is
    ranked against the archived daily series; otherwise the latest
    archived value ranks itself (legacy behaviour). Returns None when the
    panel should stay hidden. Extra keys: "start" (first history date),
    "live" (True when current_iv was supplied by the caller).
    """
    sub = ticker_history(ticker)
    if sub is None:
        return None
    ivs = sub["atm_iv"].dropna()
    if len(ivs) < MIN_POINTS:
        return None
    live = current_iv is not None and np.isfinite(current_iv)
    cur = float(current_iv) if live else float(ivs.iloc[-1])
    pct = float((ivs <= cur).mean() * 100.0)
    return {
        "current_iv": cur,
        "percentile": pct,
        "n": len(ivs),
        "as_of": sub["date"].iloc[-1].date().isoformat(),
        "start": sub["date"].iloc[0].date().isoformat(),
        "live": bool(live),
        "series": sub[["date", "atm_iv"]],
    }


def gex_history_rows(ticker: str) -> dict | None:
    """Trailing net GEX + gamma flip series for one ticker, or None."""
    sub = ticker_history(ticker)
    if sub is None:
        return None
    gex = sub[["date", "net_gex_m"]].dropna()
    if len(gex) < MIN_POINTS:
        return None
    return {
        "gex": gex,
        "flip": sub[["date", "gamma_flip"]].dropna(),
        "as_of": sub["date"].iloc[-1].date().isoformat(),
    }
