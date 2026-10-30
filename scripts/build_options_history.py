#!/usr/bin/env python3
"""Build data/options_history.csv from the daily options-chain archive.

For each (snapshot_date, ticker) in the archive parquet files, computes:
  - net_gex_m : net dealer gamma exposure in $M per 1-pt move, over the
      nearest 3 expiries. Convention (same as gex.py [gex-v5]):
      GEX = (call_OI * call_gamma - put_OI * put_gamma) * 100 * spot / 1e6
  - gamma_flip : zero-gamma (gamma flip) strike via cumulative-net-GEX
      sign change, same method as gex.gex_by_strike
  - call_wall / put_wall : strikes with max call GEX (>= spot) / min put
      GEX (<= spot)
  - atm_iv : average of call/put CBOE IV at the strike nearest spot, on the
      nearest expiry with >= 7 days to expiry
  - spot : underlying price (median of underlying_price for the date/ticker)
  - put_oi / call_oi / pc_oi : total open interest in puts / calls across
      the full chain (all expiries, OI > 0) and the put/call OI ratio.
      pc_oi feeds the USMS put/call panel; it is a level series, so read
      it against its own recent range, not an absolute 1.0 line — SPY
      puts are structurally heavy for hedging.

Reads:  <archive>/data/chains_YYYY-MM-DD.parquet
        (override with --archive DIR)
Writes: <repo>/data/options_history.csv  (overwrites; rerunnable)

Requires duckdb (pyarrow is broken in this env).
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = (Path.home() / "workspace" / "goals" /
                   "asset-mix-analysis-dashboard-and-strategy-tester" /
                   "options-chain-archive" / "data")
OUT = REPO / "data" / "options_history.csv"

COLS = ["snapshot_date", "ticker", "expiry", "option_type", "strike",
        "open_interest", "gamma", "iv", "underlying_price"]


def load_archive(archive: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(archive / "chains_*.parquet")))
    if not files:
        raise SystemExit(f"no archive parquet files in {archive}")
    parts = []
    for f in files:
        cols = ", ".join(COLS)
        parts.append(duckdb.query(
            f"SELECT {cols} FROM '{f}'").fetchdf())
    df = pd.concat(parts, ignore_index=True)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def per_ticker_day(g: pd.DataFrame) -> dict | None:
    spot = float(g["underlying_price"].median())
    if not np.isfinite(spot) or spot <= 0:
        return None
    oi_all = g[g["open_interest"].fillna(0) > 0]
    put_oi = float(oi_all.loc[oi_all["option_type"] == "put",
                              "open_interest"].sum())
    call_oi = float(oi_all.loc[oi_all["option_type"] == "call",
                               "open_interest"].sum())
    pc_oi = (round(put_oi / call_oi, 4) if call_oi > 0 else np.nan)
    df = g[(g["open_interest"].fillna(0) > 0)
           & (g["gamma"].fillna(0) > 0)].copy()
    if df.empty:
        return None
    expiries = sorted(df["expiry"].unique())[:3]
    df = df[df["expiry"].isin(expiries)]
    snap = df["snapshot_date"].iloc[0]

    df["sign"] = np.where(df["option_type"] == "call", 1.0, -1.0)
    df["gex_m"] = (df["sign"] * df["gamma"] * df["open_interest"]
                   * 100.0 * spot / 1e6)

    piv = df.pivot_table(index="strike", columns="option_type",
                         values="gex_m", aggfunc="sum").fillna(0.0)
    for c in ("call", "put"):
        if c not in piv.columns:
            piv[c] = 0.0
    piv["net_gex"] = piv["call"] + piv["put"]  # puts already negative
    piv = piv.sort_index()

    net_gex = float(piv["net_gex"].sum())

    above = piv[piv.index >= spot]
    below = piv[piv.index <= spot]
    call_wall = (float(above["call"].idxmax())
                 if not above.empty and (above["call"] > 0).any() else np.nan)
    put_wall = (float(below["put"].idxmin())
                if not below.empty and (below["put"] < 0).any() else np.nan)

    cumsum = piv["net_gex"].cumsum()
    neg = cumsum < 0
    flip = neg & (~neg.shift(-1, fill_value=True))
    hits = flip[flip].index
    gamma_flip = float(hits[0]) if len(hits) else np.nan

    # ATM IV: nearest expiry with >= 7 days to expiry (use the FULL chain,
    # not the 3-expiry GEX subset — short-dated weeklies would miss it)
    all_exp = sorted(g["expiry"].unique())
    atm_iv = np.nan
    for e in all_exp:
        d = (e.date() - snap).days
        if d < 7:
            continue
        sub = g[(g["expiry"] == e) & (g["open_interest"].fillna(0) > 0)]
        if sub.empty:
            continue
        k0 = float(sub["strike"].iloc[(sub["strike"] - spot).abs().argsort()[:1]].iloc[0])
        ivs = sub[(sub["strike"] == k0) & (sub["iv"].fillna(0) > 0)]["iv"]
        if not ivs.empty:
            atm_iv = float(ivs.mean())
            break

    return {"date": snap.isoformat(), "ticker": g["ticker"].iloc[0],
            "net_gex_m": round(net_gex, 2),
            "gamma_flip": (round(gamma_flip, 2)
                           if np.isfinite(gamma_flip) else ""),
            "call_wall": (round(call_wall, 2)
                          if np.isfinite(call_wall) else ""),
            "put_wall": (round(put_wall, 2)
                         if np.isfinite(put_wall) else ""),
            "atm_iv": round(atm_iv, 4) if np.isfinite(atm_iv) else "",
            "spot": round(spot, 2),
            "put_oi": int(put_oi), "call_oi": int(call_oi),
            "pc_oi": pc_oi if np.isfinite(pc_oi) else ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    df = load_archive(Path(args.archive))
    rows = []
    for (_, _), g in df.groupby(["snapshot_date", "ticker"], sort=True):
        r = per_ticker_day(g)
        if r:
            rows.append(r)
    out = pd.DataFrame(rows, columns=["date", "ticker", "net_gex_m",
                                      "gamma_flip", "call_wall", "put_wall",
                                      "atm_iv", "spot", "put_oi", "call_oi",
                                      "pc_oi"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_dates = out["date"].nunique()
    print(f"wrote {args.out}: {len(out)} rows, "
          f"{n_dates} dates "
          f"({out['date'].min()} -> {out['date'].max()}), "
          f"tickers {sorted(out['ticker'].unique())}")


if __name__ == "__main__":
    main()
