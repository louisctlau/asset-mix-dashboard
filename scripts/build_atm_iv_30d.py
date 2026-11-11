#!/usr/bin/env python3
"""Build data/atm_iv_30d.csv — 30-day constant-maturity ATM IV — from the
daily options-chain archive.

Design choice: 30-DAY CONSTANT-MATURITY interpolation (VIX-style), NOT the
nearest expiry. For each (snapshot_date, ticker):
  - spot = median underlying_price for the date/ticker.
  - Per expiry with DTE >= 7, ATM IV = mean of the call/put CBOE IV at the
    strike nearest spot (rows with OI > 0 and iv > 0 only).
  - Take the two expiries bracketing 30 DTE (T1 <= 30 <= T2) and interpolate
    TOTAL VARIANCE, VIX-style:
        var30 = ((T2-30)*var1*T1 + (30-T1)*var2*T2) / ((T2-T1)*30)
        IV30  = sqrt(var30)
    An expiry landing exactly on 30 DTE is used directly.
  Why interpolation: the nearest-expiry IV (what data/options_history.csv
  carries) jumps whenever the front expiry rolls or a weekly expires, so a
  backtest priced off it sees spurious vol jumps at each roll. The 30-day
  constant-maturity series is smooth across rolls and comparable across
  dates — the right vol for pricing a 30-DTE trade every roll.
  Coverage check (2026-11-11): the bracket exists on ALL 214 ticker-days
  in the archive (lo 24-30 DTE, hi 30-36 DTE), so interpolation never falls
  back in practice. The fallback below (nearest expiry >= 7 DTE,
  method="nearest") exists only so a future rebuild with thinner expiry
  coverage still produces a row instead of silently dropping the date.

Coverage limits (honest, not invented):
  - The series starts 2026-09-28 for AAPL/AMD/GOOG/META/MU/NVDA, but SPY
    only joined the archive on 2026-10-10 — no SPY IV exists before then.
  - 2026-10-19 is a permanent archive gap (no chain file that day).
  - The Strategy Tester therefore falls back to trailing 63d realised vol
    for any entry date with no archive IV (see strategy_v2.backtest,
    vol_mode="hist_iv"); the trade log marks those rows "realized
    (fallback)" so a backtest never silently prices on invented IV.

Reads:  <archive>/data/chains_YYYY-MM-DD.parquet
        (override with --archive DIR)
Writes: <repo>/data/atm_iv_30d.csv  (overwrites; rerunnable)

Requires duckdb (pyarrow is broken in this env).
"""
import argparse
import glob
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = (Path.home() / "workspace" / "goals" /
                   "asset-mix-analysis-dashboard-and-strategy-tester" /
                   "options-chain-archive" / "data")
OUT = REPO / "data" / "atm_iv_30d.csv"
TARGET_DTE = 30
MIN_DTE = 7

COLS = ["snapshot_date", "ticker", "expiry", "option_type", "strike",
        "open_interest", "iv", "underlying_price"]


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


def atm_iv_by_expiry(g: pd.DataFrame, spot: float) -> dict[int, float]:
    """{DTE: ATM IV} for expiries with DTE >= MIN_DTE with a usable ATM IV."""
    snap = g["snapshot_date"].iloc[0]
    out: dict[int, float] = {}
    for e in sorted(g["expiry"].unique()):
        d = (e.date() - snap).days
        if d < MIN_DTE:
            continue
        sub = g[(g["expiry"] == e) & (g["open_interest"].fillna(0) > 0)]
        if sub.empty:
            continue
        k0 = float(sub["strike"].iloc[
            (sub["strike"] - spot).abs().argsort()[:1]].iloc[0])
        ivs = sub[(sub["strike"] == k0) & (sub["iv"].fillna(0) > 0)]["iv"]
        if not ivs.empty:
            out[d] = float(ivs.mean())
    return out


def iv_30d(atm: dict[int, float]) -> tuple[float, int | None, int | None, str]:
    """30-day constant-maturity IV from per-expiry ATM IVs.

    Returns (iv30, dte_lo, dte_hi, method). method is "interp" when two
    expiries bracket 30 DTE, "exact" when one lands on 30 DTE, "nearest"
    when no bracket exists (fallback — never triggered on the current
    archive, where every ticker-day brackets 30 DTE).
    """
    if not atm:
        return float("nan"), None, None, "missing"
    if TARGET_DTE in atm:
        return atm[TARGET_DTE], TARGET_DTE, TARGET_DTE, "exact"
    lo = [d for d in atm if d < TARGET_DTE]
    hi = [d for d in atm if d > TARGET_DTE]
    if lo and hi:
        t1, t2 = max(lo), min(hi)
        v1, v2 = atm[t1] ** 2, atm[t2] ** 2
        var30 = ((t2 - TARGET_DTE) * v1 * t1
                 + (TARGET_DTE - t1) * v2 * t2) / ((t2 - t1) * TARGET_DTE)
        return float(np.sqrt(max(var30, 0.0))), t1, t2, "interp"
    d = min(atm, key=lambda x: abs(x - TARGET_DTE))
    return atm[d], d, d, "nearest"


def per_ticker_day(g: pd.DataFrame) -> dict | None:
    spot = float(g["underlying_price"].median())
    if not np.isfinite(spot) or spot <= 0:
        return None
    atm = atm_iv_by_expiry(g, spot)
    iv30, d1, d2, method = iv_30d(atm)
    if not np.isfinite(iv30) or iv30 <= 0:
        return None
    return {"date": g["snapshot_date"].iloc[0].isoformat(),
            "ticker": g["ticker"].iloc[0],
            "atm_iv_30d": round(iv30, 4),
            "expiry_lo": d1 if d1 is not None else "",
            "expiry_hi": d2 if d2 is not None else "",
            "spot": round(spot, 2),
            "method": method}


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
    out = pd.DataFrame(rows, columns=["date", "ticker", "atm_iv_30d",
                                      "expiry_lo", "expiry_hi", "spot",
                                      "method"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_dates = out["date"].nunique()
    print(f"wrote {args.out}: {len(out)} rows, "
          f"{n_dates} dates "
          f"({out['date'].min()} -> {out['date'].max()}), "
          f"tickers {sorted(out['ticker'].unique())}, "
          f"methods {out['method'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
