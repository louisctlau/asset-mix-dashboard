"""Black-Scholes option strategy backtester.

European-style options priced with Black-Scholes on historical underlying
prices. No early exercise, no bid/ask spread, no commissions — modeled
prices, not market prices.
"""
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# legs: (kind, abs-delta target, quantity); quantity > 0 = long, < 0 = short.
# "stock": shares held per trade (100 = covered call).
STRATEGIES = {
    "Buy & Hold": {"stock": 100, "legs": []},
    "Covered Call": {"stock": 100, "legs": [("call", 0.30, -1)]},
    "Cash-Secured Put": {"stock": 0, "legs": [("put", 0.30, -1)]},
    "Bull Call Spread": {"stock": 0,
                         "legs": [("call", 0.50, 1), ("call", 0.25, -1)]},
    "Bear Put Spread": {"stock": 0,
                        "legs": [("put", 0.50, 1), ("put", 0.25, -1)]},
    "Long Straddle": {"stock": 0,
                      "legs": [("call", 0.50, 1), ("put", 0.50, 1)]},
    "Iron Condor": {"stock": 0,
                    "legs": [("call", 0.25, -1), ("call", 0.10, 1),
                             ("put", 0.25, -1), ("put", 0.10, 1)]},
}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sig: float,
             kind: str) -> float:
    """Black-Scholes price for a European call/put."""
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    sig = max(sig, 1e-4)
    d1 = (log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    if kind == "call":
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sig: float,
             kind: str) -> float:
    """Black-Scholes delta (calls positive, puts negative)."""
    if T <= 0:
        if kind == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    sig = max(sig, 1e-4)
    d1 = (log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sqrt(T))
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1.0


def strike_for_delta(S: float, T: float, r: float, sig: float, kind: str,
                     target_abs_delta: float) -> float:
    """Find the strike whose |delta| matches the target (bisection)."""
    lo, hi = S * 0.2, S * 3.0
    want = target_abs_delta if kind == "call" else -target_abs_delta
    for _ in range(60):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sig, kind)
        if kind == "call":
            if d > want:
                lo = mid  # delta too high -> raise strike
            else:
                hi = mid
        else:
            # put delta falls as K rises; too negative -> lower the strike
            if d < want:
                hi = mid
            else:
                lo = mid
    return (lo + hi) / 2


def capital_required(S: float, legs: list, stock: int) -> float:
    """Approximate cash/margin needed to open one unit of the trade."""
    cap = stock * S
    long_prem = sum(qty * px * 100 for _, qty, _, px in legs if qty > 0)
    cap = max(cap, long_prem)
    for kind in ("call", "put"):
        shorts = [(K, px, -qty) for k2, qty, K, px in legs
                  if k2 == kind and qty < 0]
        longs = sum(qty for k2, qty, _, _ in legs if k2 == kind and qty > 0)
        naked = max(0.0, sum(n for _, _, n in shorts) - longs)
        if naked <= 0:
            continue
        if kind == "call" and stock >= 100 * naked:
            continue  # covered by shares
        for K, px, n in shorts:
            need = min(naked, n)
            cap = max(cap, (K * 100 - px * 100) * need)  # collateral less credit
            naked -= need
    return cap


def backtest(prices: pd.Series, strategy: str, dte: int, r: float,
             vol_mode: str, fixed_iv: float) -> pd.DataFrame:
    """Roll the strategy every `dte` trading days; hold each trade to expiry.

    Returns a DataFrame with one row per trade.
    """
    spec = STRATEGIES[strategy]
    stock = spec["stock"]
    log_rets = np.log(prices / prices.shift(1))
    T = dte / TRADING_DAYS
    rows = []
    i = 63  # warmup for trailing vol
    n = len(prices)
    while i + dte < n:
        S = float(prices.iloc[i])
        t_in = prices.index[i]
        sig = (fixed_iv if vol_mode == "fixed"
               else max(float(log_rets.iloc[i - 63:i].std() * sqrt(TRADING_DAYS)),
                        0.05))
        priced = []
        for kind, dd, qty in spec["legs"]:
            K = strike_for_delta(S, T, r, sig, kind, dd)
            px = bs_price(S, K, T, r, sig, kind)
            priced.append((kind, qty, K, px))
        S_T = float(prices.iloc[i + dte])
        t_out = prices.index[i + dte]
        entry_flow = (-sum(qty * px * 100 for _, qty, _, px in priced)
                      - stock * S)
        expiry_val = stock * S_T
        for kind, qty, K, _ in priced:
            intrinsic = max(S_T - K, 0.0) if kind == "call" else max(K - S_T, 0.0)
            expiry_val += qty * intrinsic * 100
        pnl = entry_flow + expiry_val
        cap = capital_required(S, priced, stock)
        leg_txt = "; ".join(
            f"{'+' if qty > 0 else ''}{qty} {kind} K={K:.1f} @ ${px:.2f}"
            for kind, qty, K, px in priced) or "—"
        rows.append({
            "Entry": t_in.date(), "S": S, "Legs": leg_txt,
            "Net premium": -sum(qty * px * 100 for _, qty, _, px in priced),
            "Expiry": t_out.date(), "S_T": S_T,
            "P&L": pnl, "Capital": cap, "IV": sig,
        })
        i += dte
    return pd.DataFrame(rows)


def trade_stats(trades: pd.DataFrame) -> dict:
    """Summary stats for a trades DataFrame (non-compounded, fixed size)."""
    if trades.empty:
        return {}
    pnl = trades["P&L"]
    c0 = float(trades["Capital"].iloc[0])
    equity = c0 + pnl.cumsum()
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    dd = equity / equity.cummax() - 1
    return {
        "Trades": len(trades),
        "Win rate": len(wins) / len(trades),
        "Total P&L": pnl.sum(),
        "Return on capital": pnl.sum() / c0 if c0 else float("nan"),
        "Avg P&L / trade": pnl.mean(),
        "Avg win": wins.mean() if len(wins) else 0.0,
        "Avg loss": losses.mean() if len(losses) else 0.0,
        "Profit factor": (-wins.sum() / losses.sum()
                          if losses.sum() < 0 else float("inf")),
        "Max drawdown": dd.min(),
        "Best trade": pnl.max(),
        "Worst trade": pnl.min(),
        "Capital (first trade)": c0,
    }
