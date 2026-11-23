#!/usr/bin/env python3
"""Build data/latest_gex_chains.csv from the daily options-chain archive.

Takes the LATEST snapshot in the archive parquet files and writes the
full chains (all expiries, all tickers) as one CSV the apps can read
from the repo's raw GitHub URL. Purpose: when Yahoo Finance rate-limits
Streamlit Cloud (HTTP 429) and CBOE is Cloudflare-blocked, the GEX
charts render from yesterday's close instead of failing — the apps
fall back to this file and label the charts "as of <date> close".

Columns: snapshot_date, ticker, expiry, option_type, strike, bid, ask,
volume, open_interest, iv, delta, gamma, theta, vega, underlying_price.
The greek columns are the archive's model/CBOE values; callers map
delta/gamma/theta/vega onto their own cboe_* names.

Reads:  <archive>/data/chains_YYYY-MM-DD.parquet
        (override with --archive DIR)
Writes: <repo>/data/latest_gex_chains.csv  (overwrites; rerunnable)

Requires duckdb (pyarrow is broken in this env).
"""
import argparse
import glob
from pathlib import Path

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = (Path.home() / "workspace" / "goals" /
                   "asset-mix-analysis-dashboard-and-strategy-tester" /
                   "options-chain-archive" / "data")
OUT = REPO / "data" / "latest_gex_chains.csv"

COLS = ["snapshot_date", "ticker", "expiry", "option_type", "strike",
        "bid", "ask", "volume", "open_interest", "iv", "delta", "gamma",
        "theta", "vega", "underlying_price"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.archive) / "chains_*.parquet")))
    if not files:
        raise SystemExit(f"no archive parquet files in {args.archive}")
    cols = ", ".join(COLS)
    df = duckdb.query(
        f"SELECT {cols} FROM '{files[-1]}'").fetchdf()
    snap = str(df["snapshot_date"].iloc[0])
    df = df.sort_values(["ticker", "expiry", "option_type",
                         "strike"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, snapshot {snap}, "
          f"tickers {sorted(df['ticker'].unique())}")


if __name__ == "__main__":
    main()
