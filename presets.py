"""One-click portfolio presets + JSON config save/load for Asset Mix.

Presets use standard definitions:
- 60/40: 60% US stocks / 40% US bonds
- All Equity: 100% stocks (large/mid/small tilt)
- Conservative: 30% stocks / 70% bonds + gold
- Golden Butterfly: 20% each — US stocks, small caps, long-term Treasuries,
  short-term Treasuries, gold (Tyler's Portfolio Charts definition)
- Permanent Portfolio: 25% each — US stocks, long-term Treasuries,
  short-term Treasuries, gold (Harry Browne definition)
"""
from __future__ import annotations

import json

import pandas as pd

PRESETS: dict[str, list[tuple[str, float]]] = {
    "60/40": [("SPY", 60.0), ("BND", 40.0)],
    "All Equity": [("SPY", 60.0), ("QQQ", 25.0), ("IWM", 15.0)],
    "Conservative": [("SPY", 30.0), ("BND", 60.0), ("GLD", 10.0)],
    "Golden Butterfly": [("SPY", 20.0), ("IWM", 20.0), ("TLT", 20.0),
                         ("SHY", 20.0), ("GLD", 20.0)],
    "Permanent Portfolio": [("SPY", 25.0), ("TLT", 25.0), ("SHY", 25.0),
                            ("GLD", 25.0)],
}


def preset_frame(name: str) -> pd.DataFrame:
    return pd.DataFrame({"Ticker": [t for t, _ in PRESETS[name]],
                         "Weight %": [w for _, w in PRESETS[name]]})


def config_to_json(holdings: pd.DataFrame, capital: float, lookback: str,
                   rebalance: str, benchmark: str) -> str:
    return json.dumps({
        "holdings": [{"ticker": str(r["Ticker"]),
                      "weight": float(r["Weight %"])}
                     for _, r in holdings.iterrows()],
        "capital": float(capital),
        "lookback": lookback,
        "rebalance": rebalance,
        "benchmark": benchmark,
    }, indent=2)


def config_from_json(text: str) -> dict:
    cfg = json.loads(text)
    hold = pd.DataFrame({"Ticker": [h["ticker"] for h in cfg["holdings"]],
                         "Weight %": [h["weight"] for h in cfg["holdings"]]})
    return {
        "holdings": hold,
        "capital": float(cfg.get("capital", 10000)),
        "lookback": str(cfg.get("lookback", "5Y")),
        "rebalance": str(cfg.get("rebalance", "Quarterly")),
        "benchmark": str(cfg.get("benchmark", "SPY")),
    }
