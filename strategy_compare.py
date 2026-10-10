"""Side-by-side comparison of all STRATEGIES on the same inputs.

Reuses strategy.backtest / strategy.trade_stats. Sharpe is computed on
per-trade returns (P&L / first-trade capital), annualized by trades-per-year;
CAGR from the equity curve over the trade window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import strategy as stg

TRADING_DAYS = 252


def compare_all(prices: pd.Series, dte: int, r: float, vol_mode: str,
                fixed_iv: float) -> pd.DataFrame:
    rows = []
    for name in stg.STRATEGIES:
        trades = stg.backtest(prices, name, dte, r, vol_mode, fixed_iv)
        if trades.empty:
            continue
        s = stg.trade_stats(trades)
        c0 = s.get("Capital (first trade)", np.nan)
        pnl = trades["P&L"]
        equity = c0 + pnl.cumsum()
        total_ret = equity.iloc[-1] / c0 - 1 if c0 else np.nan
        days = (pd.Timestamp(trades["Expiry"].iloc[-1])
                - pd.Timestamp(trades["Entry"].iloc[0])).days
        yrs = days / 365.25 if days > 0 else np.nan
        cagr = ((1 + total_ret) ** (1 / yrs) - 1
                if yrs and np.isfinite(total_ret) and (1 + total_ret) > 0
                else np.nan)
        tret = pnl / c0 if c0 else pnl * 0
        tpy = len(trades) / yrs if yrs else np.nan
        sharpe = (tret.mean() / tret.std() * np.sqrt(tpy)
                  if tret.std() and tpy else np.nan)
        rows.append({
            "Strategy": name,
            "Total return": total_ret,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Max drawdown": s.get("Max drawdown", np.nan),
            "Win rate": s.get("Win rate", np.nan),
            "# trades": s.get("Trades", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Total return", ascending=False)
    return df.reset_index(drop=True)
