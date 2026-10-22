"""Look-through exposure: aggregate underlying holdings across ETFs.

For each portfolio ticker, fetch the fund's disclosed top holdings and
sector weightings (yfinance funds_data). A single stock looks through to
itself at 100%. Everything is best-effort: tickers with no holdings data
are reported as uncovered rather than guessed at.

Coverage definition: the share of portfolio weight sitting in tickers for
which holdings/sector data was actually returned. Fund top-holdings are
top-10 disclosures, so the aggregated holdings table sums to less than
the covered weight — the gap is the funds' remaining holdings and cash,
reported separately as "unattributed".
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


SECTOR_LABELS = {
    "technology": "Technology",
    "financial_services": "Financial Services",
    "healthcare": "Healthcare",
    "consumer_cyclical": "Consumer Cyclical",
    "consumer_defensive": "Consumer Defensive",
    "communication_services": "Communication Services",
    "industrials": "Industrials",
    "energy": "Energy",
    "basic_materials": "Basic Materials",
    "realestate": "Real Estate",
    "utilities": "Utilities",
}


def _norm_weights(raw) -> dict[str, float]:
    """Normalize a {sector: weight} mapping (fractions or %) to fractions."""
    out: dict[str, float] = {}
    if raw is None:
        return out
    items = raw.items() if isinstance(raw, dict) else []
    for k, v in items:
        try:
            w = float(v)
        except (TypeError, ValueError):
            continue
        if w > 1.5:  # percent units
            w /= 100.0
        if w > 0:
            label = SECTOR_LABELS.get(str(k).strip().lower(),
                                      str(k).replace("_", " ").title())
            out[label] = out.get(label, 0.0) + w
    return out


def fetch_ticker_info(ticker: str) -> dict:
    """Holdings + sectors for one ticker.

    Returns {"kind": "fund"|"stock"|"unknown",
             "holdings": DataFrame(symbol, name, pct) | None,
             "sectors": {sector: fraction} | None,
             "sector": str | None}
    Never raises.
    """
    sym = ticker.strip().upper()
    res: dict = {"kind": "unknown", "holdings": None, "sectors": None,
                 "sector": None}
    try:
        t = yf.Ticker(sym)
    except Exception:  # noqa: BLE001
        return res
    # Fund path: top holdings + sector weightings.
    try:
        fd = t.funds_data
        th = fd.top_holdings
        if th is not None and not th.empty and "Holding Percent" in th.columns:
            h = th.reset_index()
            sym_col = h.columns[0]
            name_col = "Name" if "Name" in h.columns else sym_col
            df = pd.DataFrame({
                "symbol": h[sym_col].astype(str),
                "name": h[name_col].astype(str),
                "pct": pd.to_numeric(h["Holding Percent"],
                                     errors="coerce"),
            }).dropna(subset=["pct"])
            if not df.empty:
                res["holdings"] = df
                res["kind"] = "fund"
        sw = _norm_weights(getattr(fd, "sector_weightings", None))
        if sw:
            res["sectors"] = sw
            res["kind"] = "fund"
    except Exception:  # noqa: BLE001 - not a fund / no funds_data
        pass
    if res["kind"] == "fund":
        return res
    # Stock path: a single equity is its own holding; sector from info.
    # An ETF/fund that disclosed no holdings stays "unknown" (uncovered)
    # — it must NOT be counted as looking through to itself.
    try:
        info = t.info or {}
        res["sector"] = info.get("sector")
        if info.get("quoteType") == "EQUITY" or (
                info.get("quoteType") is None and res["sector"]):
            res["kind"] = "stock"
    except Exception:  # noqa: BLE001
        pass
    return res


def aggregate(weights: pd.Series, infos: dict[str, dict]) -> dict:
    """Combine per-ticker info into portfolio-level look-through views.

    weights: normalized target weights (fractions, index = tickers).
    infos: {ticker: fetch_ticker_info(ticker)}.
    """
    hold_rows: list[dict] = []
    sector_rows: dict[str, float] = {}
    covered_w = 0.0
    uncovered: list[str] = []
    shown_w = 0.0
    for tick, w in weights.items():
        info = infos.get(tick) or {}
        kind = info.get("kind", "unknown")
        if kind == "fund" and info.get("holdings") is not None:
            covered_w += float(w)
            for _, r in info["holdings"].iterrows():
                pw = float(w) * float(r["pct"])
                shown_w += pw
                hold_rows.append({"symbol": r["symbol"], "name": r["name"],
                                  "weight": pw})
            for sec, sw in (info.get("sectors") or {}).items():
                sector_rows[sec] = sector_rows.get(sec, 0.0) + float(w) * sw
        elif kind == "stock":
            covered_w += float(w)
            shown_w += float(w)
            hold_rows.append({"symbol": tick, "name": tick, "weight": float(w)})
            sec = info.get("sector") or "Unknown"
            sector_rows[sec] = sector_rows.get(sec, 0.0) + float(w)
        else:
            uncovered.append(tick)
    holdings = (pd.DataFrame(hold_rows)
                .groupby(["symbol", "name"], as_index=False)["weight"].sum()
                .sort_values("weight", ascending=False)
                .reset_index(drop=True)) if hold_rows else pd.DataFrame(
                    columns=["symbol", "name", "weight"])
    sectors = (pd.DataFrame(
        [{"sector": k, "weight": v} for k, v in sector_rows.items()])
        .sort_values("weight", ascending=False).reset_index(drop=True)
        if sector_rows else pd.DataFrame(columns=["sector", "weight"]))
    return {
        "holdings": holdings,
        "sectors": sectors,
        "coverage": covered_w,
        "shown": shown_w,
        "unattributed": max(covered_w - shown_w, 0.0),
        "uncovered": uncovered,
    }
