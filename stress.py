"""Named-regime stress replay for the Asset Mix tool.

Replays the current mix through fixed historical windows using the app's
own simulate() (passed in, so the engine is never forked). A ticker whose
price history starts after a window opens is EXCLUDED from that episode
and named in the footnote — weights renormalize over the covered tickers
only, never silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPISODES = [
    {"name": "COVID crash", "start": "2020-02-19", "end": "2020-03-23",
     "recovery": True},
    {"name": "2022 bear market", "start": "2022-01-03", "end": "2022-10-12",
     "recovery": False},
    {"name": "2018 Q4 selloff", "start": "2018-10-01", "end": "2018-12-24",
     "recovery": False},
]

COVER_GRACE_DAYS = 7      # first price within a week of window open counts
MIN_WINDOW_DAYS = 5       # need at least this many trading days in-window
PRE_DAYS = 14             # price history pulled before the window (returns)


def _window_growth(growth: pd.Series, start: pd.Timestamp,
                   end: pd.Timestamp) -> pd.Series | None:
    """Growth rebased to the last value on/before `start`, through `end`."""
    base = growth[growth.index <= start]
    if base.empty:
        return None
    inner = growth[(growth.index > start) & (growth.index <= end)]
    if len(inner) < MIN_WINDOW_DAYS:
        return None
    return pd.concat([base.iloc[[-1]], inner]) / float(base.iloc[-1])


def episode_results(prices: pd.DataFrame, weights: pd.Series,
                    rebalance: str, simulate) -> list[dict]:
    """One result dict per episode.

    prices: full-history closes (columns = mix tickers, may contain NaNs
    before a ticker's inception). weights: normalized target weights.
    simulate: the app's simulate(prices, target_w, rebalance) callable.
    """
    out: list[dict] = []
    for ep in EPISODES:
        start = pd.Timestamp(ep["start"])
        end = pd.Timestamp(ep["end"])
        covered, excluded = [], []
        for t in weights.index:
            s = prices[t].dropna() if t in prices.columns else pd.Series(
                dtype=float)
            if not s.empty and s.index[0] <= start + pd.Timedelta(
                    days=COVER_GRACE_DAYS):
                covered.append(t)
            else:
                excluded.append(t)
        res: dict = {"name": ep["name"], "start": start, "end": end,
                     "excluded": excluded, "ok": False, "return": np.nan,
                     "max_dd": np.nan, "recovery_days": None,
                     "recovery_date": None, "note": None}
        if not covered:
            res["note"] = "No holding has price history in this window."
            out.append(res)
            continue
        sub_w = weights[covered]
        lo = start - pd.Timedelta(days=PRE_DAYS)
        if ep["recovery"]:
            # COVID: run to the end of available data to time the recovery.
            seg = prices.loc[lo:, covered].dropna()
        else:
            seg = prices.loc[lo:end, covered].dropna()
        if len(seg) < MIN_WINDOW_DAYS + 1:
            res["note"] = "Not enough price history in this window."
            out.append(res)
            continue
        rets, _ = simulate(seg, sub_w, rebalance)
        growth = (1 + rets).cumprod()
        wg = _window_growth(growth, start, end)
        if wg is None:
            res["note"] = "Not enough trading days inside the window."
            out.append(res)
            continue
        res["return"] = float(wg.iloc[-1] - 1)
        res["max_dd"] = float((wg / wg.cummax() - 1).min())
        res["ok"] = True
        if ep["recovery"]:
            peak_val = float(growth[growth.index <= start].iloc[-1])
            after = growth[growth.index > end]
            rec = after[after >= peak_val]
            if not rec.empty:
                rec_date = rec.index[0]
                res["recovery_days"] = int((rec_date - start).days)
                res["recovery_date"] = rec_date.date().isoformat()
        if excluded:
            firsts = {t: (prices[t].dropna().index[0].date().isoformat()
                          if t in prices.columns
                          and not prices[t].dropna().empty else "n/a")
                      for t in excluded}
            res["note"] = ("Excluded (history starts later): "
                            + ", ".join(f"{t} ({firsts[t]})"
                                        for t in excluded)
                            + ". Weights renormalized over the rest.")
        out.append(res)
    return out
