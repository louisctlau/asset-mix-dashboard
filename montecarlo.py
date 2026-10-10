"""Monte Carlo forward simulation for a hypothetical portfolio.

Resamples the portfolio's own historical monthly returns with replacement
(bootstrapped monthly steps, 12 months, 2000 paths). Buy-and-hold is assumed
within each path — the selected rebalance rule is NOT modeled; the chart
labels this assumption. No fees, taxes, or contributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_PATHS = 2000
MONTHS = 12
SEED = 42


def simulate(monthly_rets: pd.Series, n_paths: int = N_PATHS,
             months: int = MONTHS, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame of simulated growth paths (each starting at 1.0).

    Columns are paths, index 0..months.
    """
    r = monthly_rets.dropna().values
    if len(r) < 12:
        raise ValueError("need at least 12 months of history")
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n_paths, months))
    paths = np.cumprod(1.0 + draws, axis=1)
    paths = np.hstack([np.ones((n_paths, 1)), paths]).T
    return pd.DataFrame(paths)


def bands(paths: pd.DataFrame) -> pd.DataFrame:
    """10th / 50th / 90th percentile bands across paths."""
    q = paths.quantile([0.10, 0.50, 0.90], axis=1).T
    q.columns = ["p10", "p50", "p90"]
    return q


def ending_stats(paths: pd.DataFrame, capital: float) -> dict:
    end = paths.iloc[-1] * capital
    return {
        "p10": float(end.quantile(0.10)),
        "p50": float(end.quantile(0.50)),
        "p90": float(end.quantile(0.90)),
        "prob_profit": float((end > capital).mean()),
    }
