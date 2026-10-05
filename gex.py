"""Gamma exposure (GEX) — same convention as the US Market Sentiment app.

    GEX(strike) = (call_OI * call_gamma - put_OI * put_gamma) * 100 * spot

in $M of delta-hedge flow per 1-pt move, summed over the nearest 3
expiries. Dealer positioning assumption (standard): dealers are long
calls / short puts, so calls add positive exposure and puts negative.
Positive net GEX = dealers long gamma (dampens moves, pinning);
negative = dealers short gamma (amplifies moves). [gex-v5]

Gamma input is CBOE's listed per-contract gamma (same source as USMS);
open interest is prior-day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def gex_by_strike(chain: pd.DataFrame, spot: float,
                  n_expiries: int = 3) -> dict:
    """Aggregate GEX by strike over the nearest `n_expiries`."""
    df = chain[(chain["open_interest"].fillna(0) > 0)
               & (chain["cboe_gamma"].fillna(0) > 0)].copy()
    if df.empty:
        raise RuntimeError("no usable OI/gamma in chain")
    expiries = sorted(df["expiry"].unique())[:n_expiries]
    df = df[df["expiry"].isin(expiries)]
    df["side"] = np.where(df["option_type"] == "call", "calls", "puts")
    df["sign"] = np.where(df["option_type"] == "call", 1.0, -1.0)
    df["gex_m"] = (df["sign"] * df["cboe_gamma"] * df["open_interest"]
                   * 100.0 * spot / 1e6)

    piv = df.pivot_table(index="strike", columns="side", values="gex_m",
                         aggfunc="sum").fillna(0.0)
    for c in ("calls", "puts"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv["net_gex"] = piv["calls"] + piv["puts"]  # puts already negative
    piv = piv.sort_index()

    total_net = float(piv["net_gex"].sum())
    above = piv[piv.index >= spot]
    below = piv[piv.index <= spot]
    call_wall = (float(above["calls"].idxmax())
                 if not above.empty and (above["calls"] > 0).any() else None)
    put_wall = (float(below["puts"].idxmin())
                if not below.empty and (below["puts"] < 0).any() else None)
    cumsum = piv["net_gex"].cumsum()
    neg = cumsum < 0
    flip = neg & (~neg.shift(-1, fill_value=True))
    hits = flip[flip].index
    zero_gamma = float(hits[0]) if len(hits) else None

    return {"spot": spot, "expiries": expiries,
            "strikes": piv.reset_index(), "total_net": total_net,
            "call_wall": call_wall, "put_wall": put_wall,
            "zero_gamma": zero_gamma}


def gex_chart(g: dict, title: str) -> go.Figure:
    """Net GEX by strike ($M per 1-pt move): call gamma up, put gamma down.
    Spot, walls, zero-gamma overlaid."""
    df = g["strikes"]
    spot = g["spot"]
    df = df[(df["strike"] >= spot * 0.92) & (df["strike"] <= spot * 1.08)]
    y = df["net_gex"]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in y]
    fig = go.Figure(go.Bar(x=df["strike"], y=y, marker_color=colors,
                           name="GEX by strike ($M/pt)"))
    fig.add_vline(x=spot, line_color="#1f77b4", line_width=2,
                  annotation_text=f"Spot {spot:.0f}")
    if g["put_wall"]:
        fig.add_vline(x=g["put_wall"], line_color="#2ecc71", line_dash="dash",
                      annotation_text=f"Put wall {g['put_wall']:.0f}")
    if g["call_wall"]:
        fig.add_vline(x=g["call_wall"], line_color="#e74c3c", line_dash="dash",
                      annotation_text=f"Call wall {g['call_wall']:.0f}")
    if g["zero_gamma"]:
        fig.add_vline(x=g["zero_gamma"], line_color="#f1c40f", line_dash="dot",
                      annotation_text=f"γflip {g['zero_gamma']:.0f}")
    fig.update_layout(title=title, height=380, margin=dict(t=40, b=10),
                      xaxis_title="Strike",
                      yaxis_title="Net GEX ($M per 1-pt move)",
                      bargap=0.1)
    return fig


def gex_read(g: dict) -> str:
    """One-line interpretation of the positioning."""
    t = g["total_net"]
    tone = ("dealers long gamma — moves tend to be dampened/pinned"
            if t > 0 else
            "dealers short gamma — moves tend to be amplified")
    zg = (f" γflip at {g['zero_gamma']:.0f}: below it, dealers are "
          f"short gamma and moves can accelerate." if g["zero_gamma"] else "")
    return f"Net GEX ${t:+.0f}M/pt — {tone}.{zg}"
