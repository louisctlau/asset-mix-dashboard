"""Volatility surface analytics from a CBOE chain DataFrame.

- smile: listed IV vs strike per expiry (calls and puts separately)
- skew: 25-delta risk reversal and butterfly per expiry
- term structure: ATM IV across expiries
- chain_greeks: model greeks (greeks.py) for every quoted contract

IVs are CBOE's listed values; greeks are recomputed with the
Black-Scholes engine so vanna/charm (not published by CBOE) sit
alongside delta/gamma/theta on one consistent model.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import greeks as gk

_IV_CAP = 5.0  # ignore listed IV prints above 500%


def _quoted(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable two-sided quote and sane listed IV."""
    q = df[(df["iv"].notna()) & (df["iv"] > 0) & (df["iv"] < _IV_CAP)].copy()
    q = q[q["bid"].fillna(0) > 0]
    return q


def expiries(chain: pd.DataFrame) -> list:
    """Sorted unique expiries present in the chain."""
    return sorted(chain["expiry"].dropna().unique().tolist())


def dte(expiry, today: date | None = None) -> int:
    today = today or date.today()
    return max((pd.Timestamp(expiry).date() - today).days, 0)


def smile(chain: pd.DataFrame, expiry) -> pd.DataFrame:
    """Per-strike listed IV for one expiry: strike, call_iv, put_iv."""
    q = _quoted(chain[chain["expiry"] == expiry])
    calls = q[q["option_type"] == "call"].groupby("strike")["iv"].median()
    puts = q[q["option_type"] == "put"].groupby("strike")["iv"].median()
    out = pd.DataFrame({"call_iv": calls, "put_iv": puts})
    out.index.name = "strike"
    return out.reset_index().sort_values("strike")


def atm_iv(chain: pd.DataFrame, expiry, S: float) -> float | None:
    """ATM listed IV: average of call/put IV at the strike nearest spot."""
    sm = smile(chain, expiry).dropna()
    if sm.empty:
        return None
    row = sm.iloc[(sm["strike"] - S).abs().argsort()[:1]]
    ivs = [v for v in (row["call_iv"].iloc[0], row["put_iv"].iloc[0])
           if pd.notna(v)]
    return float(np.mean(ivs)) if ivs else None


def _iv_at_delta(chain: pd.DataFrame, expiry, kind: str,
                 target_abs_delta: float) -> float | None:
    """Listed IV at the strike where |delta| == target (linear interp)."""
    q = _quoted(chain[(chain["expiry"] == expiry)
                      & (chain["option_type"] == kind)
                      & (chain["cboe_delta"].notna())])
    if q.empty:
        return None
    q = q.sort_values("strike")
    d = q["cboe_delta"].abs().to_numpy()
    iv = q["iv"].to_numpy()
    tgt = target_abs_delta
    if not (d.min() <= tgt <= d.max()):
        return None
    # |delta| falls as strike rises for calls, rises for puts — interp on
    # the monotonic-in-|delta| ordering via searchsorted on sorted |delta|.
    order = np.argsort(d)
    ds, ivs = d[order], iv[order]
    return float(np.interp(tgt, ds, ivs))


def skew(chain: pd.DataFrame, expiry, S: float) -> dict:
    """25-delta risk reversal / butterfly and ATM IV for one expiry.

    risk_reversal_25 = IV(25d call) - IV(25d put): > 0 means upside calls
    are bid vs downside puts (call skew); < 0 the usual equity put skew.
    butterfly_25 = avg(25d wings) - ATM IV: convexity of the smile.
    """
    atm = atm_iv(chain, expiry, S)
    c25 = _iv_at_delta(chain, expiry, "call", 0.25)
    p25 = _iv_at_delta(chain, expiry, "put", 0.25)
    rr = (c25 - p25) if c25 is not None and p25 is not None else None
    bf = ((c25 + p25) / 2 - atm) if rr is not None and atm else None
    return {"atm_iv": atm, "call_25d_iv": c25, "put_25d_iv": p25,
            "risk_reversal_25": rr, "butterfly_25": bf}


def term_structure(chain: pd.DataFrame, S: float,
                   max_expiries: int = 12, min_dte: int = 1) -> pd.DataFrame:
    """ATM IV, 25d RR and 25d BF per expiry (nearest `max_expiries`).

    Skips expiries expiring sooner than `min_dte` days — 0DTE smiles are
    degenerate and add noise to the surface.
    """
    rows = []
    for exp in expiries(chain)[:max_expiries]:
        if dte(exp) < min_dte:
            continue
        sk = skew(chain, exp, S)
        rows.append({"expiry": exp, "dte": dte(exp), **sk})
    return pd.DataFrame(rows)


def chain_greeks(chain: pd.DataFrame, expiry, S: float, r: float,
                 q: float) -> pd.DataFrame:
    """Quoted contracts for one expiry with model greeks attached."""
    today = date.today()
    rows = _quoted(chain[chain["expiry"] == expiry])
    out = []
    for _, row in rows.iterrows():
        T = max(dte(expiry, today), 0.5) / 365.0  # floor: half a day
        g = gk.greeks(S, float(row["strike"]), T, r, q, float(row["iv"]),
                      row["option_type"])
        out.append({
            "strike": row["strike"], "type": row["option_type"],
            "iv": row["iv"], "bid": row["bid"], "ask": row["ask"],
            "volume": row["volume"], "open_interest": row["open_interest"],
            **{k: g[k] for k in ("delta", "gamma", "theta", "vega",
                                 "vanna", "charm")},
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["strike", "type"]).reset_index(drop=True)
    return df
