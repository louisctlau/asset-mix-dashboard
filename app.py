"""Portfolio Lab — hypothetical asset-mix analysis + Black-Scholes strategy tester.

Asset Mix: define a hypothetical portfolio (tickers + target weights) and see
what it would have done on Yahoo Finance total-return data.

Strategy Tester: backtest stock/option strategies on historical prices with
options priced by Black-Scholes (European, no early exercise, no bid/ask).

Options Analytics: greeks (delta, gamma, theta, vanna, charm) and the
volatility surface (smile, skew, term structure) on the live CBOE delayed
option chain.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import option_chains as chains
import gex
import strategy as stg
import volsurface as vol
import glossary as gl
import iv_history as ivh
import montecarlo as mc
import presets
import strategy_compare as scmp

st.set_page_config(page_title="Portfolio Lab", layout="wide")

PERIODS = {"1Y": "1y", "3Y": "3y", "5Y": "5y", "10Y": "10y", "Max": "max"}
TRADING_DAYS = 252

DISCLAIMER = "For educational and experimental purposes only. Not investment advice."


def show_disclaimer():
    st.markdown(
        '<p style="color:#ff4b4b; font-size:0.85rem;">' + DISCLAIMER + "</p>",
        unsafe_allow_html=True,
    )


DEFAULT_HOLDINGS = pd.DataFrame(
    {"Ticker": ["SPY", "QQQ", "BND", "GLD"], "Weight %": [50.0, 10.0, 30.0, 10.0]}
)

st.sidebar.header("Tool")
tool = st.sidebar.radio("Tool",
                        ["Asset Mix", "Strategy Tester", "Options Analytics"],
                        label_visibility="collapsed")


@st.cache_data(ttl=86400, show_spinner=False)
def load_closes(tickers: tuple, period: str) -> pd.DataFrame:
    """Adjusted closes (dividends reinvested) for the given tickers.

    Raises RuntimeError if Yahoo returns nothing for any requested ticker.
    Partial frames are never returned, so a transient single-ticker gap can
    never be cached and replayed for the full 24h TTL (exceptions raised
    here are not cached by st.cache_data — the next visit retries fresh).
    """
    last_err = None
    data = None
    for _ in range(3):
        try:
            data = yf.download(list(tickers), period=period, auto_adjust=True,
                               progress=False)
            break
        except Exception as e:  # transient Yahoo failure; retry
            last_err = e
    else:
        raise RuntimeError(f"Yahoo Finance fetch failed: {last_err}")
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:  # single ticker
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})
    closes = closes.dropna(how="all")
    missing = [t for t in tickers
               if t not in closes.columns or closes[t].dropna().empty]
    if missing:
        raise RuntimeError(
            f"Yahoo Finance returned no data for: {', '.join(missing)}")
    return closes


@st.cache_data(ttl=900, show_spinner=False)
def load_chain_bundle(ticker: str):
    """Live CBOE delayed chain + underlying + fetch time (15-min cache).

    Raises RuntimeError on fetch failure or an empty chain (never cached,
    so the next visit retries fresh).
    """
    cboe_sym, _ = chains.resolve_ticker(ticker)
    chain, spot = chains.fetch_chain(cboe_sym)
    if chain.empty or spot is None:
        raise RuntimeError(f"No chain data for {ticker}")
    return chain, spot, datetime.now(timezone.utc)


@st.cache_data(ttl=86400, show_spinner=False)
def risk_free_rate() -> float:
    """3M T-bill yield (^IRX) as a fraction; 4% on any failure."""
    try:
        hist = yf.Ticker("^IRX").history(period="5d")
        if not hist.empty:
            y = float(hist["Close"].iloc[-1]) / 100.0
            if 0.0 < y < 0.25:
                return y
    except Exception:
        pass
    return 0.04


@st.cache_data(ttl=86400, show_spinner=False)
def div_yield(ticker: str) -> float:
    """Trailing dividend yield as a fraction; 0 on any failure.

    Uses trailingAnnualDividendYield (a true ratio). Note: Yahoo's
    `dividendYield` field is in percent units (e.g. 0.43 for NVDA) and
    must NOT be used directly. Cash indices (^SPX etc.) publish no yield
    field, so q=0 for them (negligible except on long-dated delta).
    """
    try:
        _, yahoo_sym = chains.resolve_ticker(ticker)
        y = yf.Ticker(yahoo_sym).info.get("trailingAnnualDividendYield") or 0.0
        return min(max(float(y), 0.0), 0.25)
    except Exception:
        return 0.0


def simulate(prices: pd.DataFrame, target_w: pd.Series,
             rebalance: str) -> tuple[pd.Series, pd.DataFrame]:
    """Backtest a target-weight portfolio.

    Returns (daily portfolio returns, DataFrame of evolving weights).
    """
    rets = prices.pct_change().dropna()
    if rebalance == "Monthly":
        marks = set(rets.resample("ME").last().index.date)
    elif rebalance == "Quarterly":
        marks = set(rets.resample("QE").last().index.date)
    elif rebalance == "Annual":
        marks = set(rets.resample("YE").last().index.date)
    else:
        marks = set()

    w = target_w / target_w.sum()
    cur = w.copy()
    port_rets, weight_rows = [], []
    for dt, r in rets.iterrows():
        if dt.date() in marks:
            cur = w.copy()
        port_rets.append(float((cur * r).sum()))
        cur = cur * (1 + r)
        cur = cur / cur.sum()
        weight_rows.append(cur.copy())
    port = pd.Series(port_rets, index=rets.index, name="portfolio")
    whist = pd.DataFrame(weight_rows, index=rets.index)
    return port, whist


def mix_stats(growth: pd.Series, port_rets: pd.Series) -> dict:
    n = len(port_rets)
    yrs = n / TRADING_DAYS if n else np.nan
    total = growth.iloc[-1] / growth.iloc[0] - 1
    cagr = (growth.iloc[-1] / growth.iloc[0]) ** (1 / yrs) - 1 if yrs else np.nan
    vol = port_rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (port_rets.mean() / port_rets.std() * np.sqrt(TRADING_DAYS)
              if port_rets.std() else np.nan)
    dd = growth / growth.cummax() - 1
    yearly = growth.resample("YE").last().pct_change().dropna()
    return {
        "Total return": total,
        "CAGR": cagr,
        "Volatility (ann.)": vol,
        "Sharpe (rf=0)": sharpe,
        "Max drawdown": dd.min(),
        "Best year": yearly.max() if len(yearly) else np.nan,
        "Worst year": yearly.min() if len(yearly) else np.nan,
    }


# ============================ ASSET MIX ============================
if tool == "Asset Mix":
    st.title("Asset Mix Dashboard")
    st.caption("Hypothetical portfolio — what would this asset mix have done?")
    show_disclaimer()
    gl.show_glossary("Asset Mix")

    st.sidebar.header("Hypothetical portfolio")
    with st.sidebar.expander("Presets & saved mixes", expanded=False):
        cols = st.columns(3)
        for i, name in enumerate(presets.PRESETS):
            if cols[i % 3].button(name, key=f"preset_{i}",
                                  use_container_width=True):
                st.session_state["holdings"] = presets.preset_frame(name)
        up = st.file_uploader("Load config (JSON)", type="json",
                              key="cfg_upload")
        if up is not None and st.session_state.get("_cfg_loaded") != up.name:
            try:
                cfg = presets.config_from_json(
                    up.read().decode("utf-8"))
                st.session_state["holdings"] = cfg["holdings"]
                st.session_state["capital"] = cfg["capital"]
                for k, allowed in (
                        ("lookback", list(PERIODS)),
                        ("rebalance", ["Monthly", "Quarterly", "Annual",
                                       "None"]),
                        ("benchmark", ["SPY", "60/40 (SPY/BND)", "None"])):
                    if cfg[k] in allowed:
                        st.session_state[k] = cfg[k]
                st.session_state["_cfg_loaded"] = up.name
            except Exception:
                st.warning("Could not read that config file.")
    edited = st.sidebar.data_editor(
        DEFAULT_HOLDINGS, num_rows="dynamic", use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", max_chars=10),
            "Weight %": st.column_config.NumberColumn("Weight %", min_value=0,
                                                      max_value=100, step=5),
        },
        key="holdings",
    )
    capital = st.sidebar.number_input("Initial capital ($)", min_value=100,
                                      value=10000, step=1000, key="capital")
    lookback = st.sidebar.selectbox("Lookback", list(PERIODS), index=2,
                                    key="lookback")
    rebalance = st.sidebar.selectbox(
        "Rebalance", ["Monthly", "Quarterly", "Annual", "None"], index=1,
        key="rebalance")
    benchmark = st.sidebar.selectbox(
        "Benchmark", ["SPY", "60/40 (SPY/BND)", "None"], index=0,
        key="benchmark")

    hold = edited.copy()
    hold["Ticker"] = hold["Ticker"].astype(str).str.strip().str.upper()
    hold = hold[(hold["Ticker"] != "") & (hold["Ticker"] != "NAN")]
    hold["Weight %"] = pd.to_numeric(hold["Weight %"], errors="coerce").fillna(0)
    hold = hold[hold["Weight %"] > 0]
    if hold.empty:
        st.warning("Add at least one holding with a positive weight.")
        st.stop()

    tickers = hold["Ticker"].tolist()
    weights = pd.Series(hold["Weight %"].values, index=tickers, dtype=float)
    if abs(weights.sum() - 100) > 1e-9:
        st.sidebar.caption(
            f"Weights sum to {weights.sum():.1f}% — normalized to 100%.")
    weights = weights / weights.sum()
    st.sidebar.download_button(
        "Download config (JSON)",
        presets.config_to_json(hold, capital, lookback, rebalance,
                               benchmark),
        file_name="portfolio_config.json", mime="application/json")

    needed = set(tickers)
    if benchmark == "60/40 (SPY/BND)":
        needed |= {"SPY", "BND"}
    elif benchmark == "SPY":
        needed.add("SPY")

    with st.spinner("Fetching market data…"):
        try:
            closes = load_closes(tuple(sorted(needed)), PERIODS[lookback])
        except RuntimeError:
            # load_closes raises on total failure or a partial frame (the
            # latter is never cached, so the next visit retries fresh).
            st.error("Market data is temporarily unavailable — please try "
                     "again in a minute.")
            st.stop()

    closes = closes.dropna()
    if closes.empty:
        st.error("Market data is temporarily unavailable — please try again "
                 "in a minute.")
        st.stop()

    prices = closes[[t for t in weights.index]]
    port_rets, whist = simulate(prices, weights, rebalance)
    growth = (1 + port_rets).cumprod() * capital

    bench_series = {}
    if benchmark == "SPY" and "SPY" in closes.columns:
        spy_rets = closes["SPY"].pct_change().dropna()
        spy_rets = spy_rets.reindex(port_rets.index).dropna()
        bench_series["SPY"] = (1 + spy_rets).cumprod() * capital
    elif benchmark == "60/40 (SPY/BND)" and {"SPY", "BND"} <= set(closes.columns):
        w6040 = pd.Series([0.6, 0.4], index=["SPY", "BND"])
        b_rets, _ = simulate(closes[["SPY", "BND"]].dropna(), w6040, "Monthly")
        b_rets = b_rets.reindex(port_rets.index).dropna()
        bench_series["60/40"] = (1 + b_rets).cumprod() * capital

    s = mix_stats(growth, port_rets)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{s['Total return']:.1%}")
    c2.metric("CAGR", f"{s['CAGR']:.1%}")
    c3.metric("Volatility", f"{s['Volatility (ann.)']:.1%}")
    c4.metric("Max drawdown", f"{s['Max drawdown']:.1%}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe (rf=0)", f"{s['Sharpe (rf=0)']:.2f}")
    c2.metric("Best year", f"{s['Best year']:.1%}")
    c3.metric("Worst year", f"{s['Worst year']:.1%}")
    c4.metric("Holdings", f"{len(weights)}")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Target allocation")
        fig = px.pie(names=weights.index, values=weights.values, hole=0.45)
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10),
                          height=320)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader(f"Growth of ${capital:,}")
        gfig = go.Figure()
        gfig.add_trace(go.Scatter(x=growth.index, y=growth.values,
                                  name="Portfolio", line=dict(width=2.5)))
        for name, bs in bench_series.items():
            gfig.add_trace(go.Scatter(x=bs.index, y=bs.values, name=name,
                                      line=dict(dash="dash")))
        gfig.update_layout(yaxis_title="$", margin=dict(t=10, b=10, l=10, r=10),
                           height=320, legend=dict(orientation="h", y=1.05))
        st.plotly_chart(gfig, use_container_width=True)

    st.subheader("Drawdown")
    dd = growth / growth.cummax() - 1
    dfig = px.area(x=dd.index, y=dd.values, labels={"x": "", "y": ""})
    dfig.update_traces(fillcolor="rgba(239,68,68,0.25)",
                       line=dict(color="rgb(239,68,68)"))
    dfig.update_layout(yaxis_tickformat=".0%", height=240,
                       margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(dfig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation of holdings")
        corr = prices.pct_change().corr()
        hfig = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        hfig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(hfig, use_container_width=True)
    with c2:
        st.subheader("Weight drift (latest vs target)")
        drift = pd.DataFrame({
            "Target %": weights * 100,
            "Current %": whist.iloc[-1] * 100,
        })
        drift["Drift (pp)"] = drift["Current %"] - drift["Target %"]
        drift = drift.round(1)
        st.dataframe(drift.sort_values("Drift (pp)", key=abs, ascending=False),
                     use_container_width=True)
        st.caption(f"Rebalancing: {rebalance.lower()} · "
                   f"{growth.index[0].date()} → {growth.index[-1].date()}")

    st.subheader("Rolling 1-year risk metrics")
    st.caption("Each stat recomputed on a trailing 252-trading-day window.")
    roll = port_rets.rolling(TRADING_DAYS)
    roll_sharpe = roll.mean() / roll.std() * np.sqrt(TRADING_DAYS)
    roll_vol = roll.std() * np.sqrt(TRADING_DAYS)
    roll_dd = growth.rolling(TRADING_DAYS).apply(
        lambda w: float((w / w.cummax() - 1).min()), raw=False)
    bench_roll = {}
    for name, bs in bench_series.items():
        br = bs.pct_change().dropna()
        broll = br.rolling(TRADING_DAYS)
        bg = bs
        bench_roll[name] = (
            broll.mean() / broll.std() * np.sqrt(TRADING_DAYS),
            broll.std() * np.sqrt(TRADING_DAYS),
            bg.rolling(TRADING_DAYS).apply(
                lambda w: float((w / w.cummax() - 1).min()), raw=False),
        )
    for title, pser, bidx, fmt in (
            ("Sharpe", roll_sharpe, 0, ".2f"),
            ("Volatility (ann.)", roll_vol, 1, ".1%"),
            ("Max drawdown", roll_dd, 2, ".1%")):
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=pser.index, y=pser.values,
                                  name="Portfolio", line=dict(width=2.5)))
        for name, bro in bench_roll.items():
            rfig.add_trace(go.Scatter(x=bro[bidx].index, y=bro[bidx].values,
                                      name=name, line=dict(dash="dash",
                                                           width=1.2,
                                                           color="rgba(160,160,160,0.7)")))
        rfig.update_layout(title=title, yaxis_tickformat=fmt, height=260,
                           margin=dict(t=40, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(rfig, use_container_width=True)

    st.subheader("Monte Carlo — 1 year forward")
    monthly = port_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    if len(monthly) < 12:
        st.caption("Not enough history for a 12-month simulation — "
                   "pick a longer lookback.")
    else:
        paths = mc.simulate(monthly)
        b = mc.bands(paths)
        mstats = mc.ending_stats(paths, capital)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median outcome", f"${mstats['p50']:,.0f}")
        m2.metric("10th percentile", f"${mstats['p10']:,.0f}")
        m3.metric("90th percentile", f"${mstats['p90']:,.0f}")
        m4.metric("Chance of profit", f"{mstats['prob_profit']:.0%}")
        mfig = go.Figure()
        mfig.add_trace(go.Scatter(x=b.index, y=b["p90"] * capital,
                                  line=dict(width=0), showlegend=False,
                                  hoverinfo="skip"))
        mfig.add_trace(go.Scatter(x=b.index, y=b["p10"] * capital,
                                  line=dict(width=0), showlegend=False,
                                  hoverinfo="skip",
                                  fill="tonexty",
                                  fillcolor="rgba(99,110,250,0.20)"))
        sample = paths.sample(min(20, paths.shape[1]), axis=1, random_state=7)
        for col in sample.columns:
            mfig.add_trace(go.Scatter(x=sample.index,
                                      y=sample[col] * capital,
                                      line=dict(width=0.7,
                                                color="rgba(150,150,150,0.35)"),
                                      showlegend=False, hoverinfo="skip"))
        mfig.add_trace(go.Scatter(x=b.index, y=b["p50"] * capital,
                                  name="Median path",
                                  line=dict(width=2.5, color="#636EFA")))
        mfig.update_layout(yaxis_title="$", xaxis_title="Months ahead",
                           height=360, margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(mfig, use_container_width=True)
        st.caption("2,000 paths; monthly steps resampled with replacement "
                   "from this portfolio's own history. Buy-and-hold assumed "
                   "within each path (rebalance not modeled); no fees, "
                   "taxes, or contributions. A planning range, not a "
                   "prediction.")

    gdf = pd.DataFrame({"Portfolio": growth})
    for name, bs in bench_series.items():
        gdf[name] = bs
    st.download_button("Download growth series (CSV)", gdf.to_csv(),
                       "growth_series.csv", "text/csv")

    st.caption("Data: Yahoo Finance adjusted closes (dividends reinvested), "
               "daily. Hypothetical backtest — not investment advice.")

# ========================== STRATEGY TESTER ==========================
elif tool == "Strategy Tester":
    st.title("Strategy Tester")
    st.caption("Black-Scholes backtest — modeled option prices on historical "
               "stock data.")
    show_disclaimer()
    gl.show_glossary("Strategy Tester")

    st.sidebar.header("Strategy")
    ticker = st.sidebar.text_input("Ticker", value="SPY").strip().upper()
    strat_name = st.sidebar.selectbox("Strategy", list(stg.STRATEGIES))
    dte = st.sidebar.number_input("Days to expiry", min_value=7, max_value=365,
                                  value=30, step=5)
    lookback = st.sidebar.selectbox("Lookback", list(PERIODS), index=2)
    vol_mode = st.sidebar.radio("Volatility",
                                ["Trailing 63d realized", "Fixed IV"],
                                index=0)
    fixed_iv = st.sidebar.number_input("Fixed IV %", min_value=1.0,
                                       max_value=200.0, value=25.0,
                                       step=1.0) / 100.0
    rf = st.sidebar.number_input("Risk-free rate %", min_value=0.0,
                                 max_value=20.0, value=4.0, step=0.25) / 100.0

    if not ticker:
        st.warning("Enter a ticker.")
        st.stop()

    with st.spinner("Fetching market data…"):
        try:
            closes = load_closes((ticker,), PERIODS[lookback])
        except RuntimeError:
            st.error(f"Couldn't fetch data for {ticker} — please try again "
                     "in a minute.")
            st.stop()
    prices = closes[ticker].dropna()

    mode = "fixed" if vol_mode == "Fixed IV" else "realized"
    with st.spinner("Running backtest…"):
        trades = stg.backtest(prices, strat_name, int(dte), rf, mode, fixed_iv)
    if trades.empty:
        st.warning("Not enough history for these settings.")
        st.stop()
    s = stg.trade_stats(trades)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", f"{s['Trades']}")
    c2.metric("Win rate", f"{s['Win rate']:.0%}")
    c3.metric("Total P&L", f"${s['Total P&L']:,.0f}")
    c4.metric("Return on capital", f"{s['Return on capital']:.1%}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg P&L / trade", f"${s['Avg P&L / trade']:,.0f}")
    c2.metric("Profit factor",
              f"{s['Profit factor']:.2f}" if np.isfinite(s['Profit factor'])
              else "∞")
    c3.metric("Max drawdown", f"{s['Max drawdown']:.1%}")
    c4.metric("Capital (first trade)", f"${s['Capital (first trade)']:,.0f}")

    st.subheader("Equity curve")
    c0 = s["Capital (first trade)"]
    equity = c0 + trades.set_index("Expiry")["P&L"].cumsum()
    first_px = prices.loc[prices.index <= pd.Timestamp(trades["Entry"].iloc[0])].iloc[-1]
    bh = c0 * prices / first_px
    bh = bh[bh.index >= pd.Timestamp(trades["Entry"].iloc[0])]
    efig = go.Figure()
    efig.add_trace(go.Scatter(x=equity.index, y=equity.values,
                              name=strat_name, line=dict(width=2.5)))
    efig.add_trace(go.Scatter(x=bh.index, y=bh.values, name=f"Buy & Hold {ticker}",
                              line=dict(dash="dash")))
    efig.update_layout(yaxis_title="$", height=360,
                       margin=dict(t=10, b=10, l=10, r=10),
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(efig, use_container_width=True)

    st.subheader("All strategies — side by side")
    st.caption("Same ticker, lookback, expiry and volatility settings for "
               "every strategy.")
    with st.spinner("Running all strategies…"):
        comp = scmp.compare_all(prices, int(dte), rf, mode, fixed_iv)
    if not comp.empty:
        disp = comp.copy()
        for pc in ["Total return", "CAGR", "Max drawdown", "Win rate"]:
            disp[pc] = (disp[pc] * 100).round(1).astype(str) + "%"
        disp["Sharpe"] = disp["Sharpe"].round(2)
        st.dataframe(disp, use_container_width=True)
        st.download_button("Download comparison (CSV)",
                           comp.to_csv(index=False),
                           "strategy_comparison.csv", "text/csv")

    with st.expander(f"Trade log ({len(trades)} trades)"):
        show = trades.copy()
        for col in ["S", "S_T", "Net premium", "P&L", "Capital"]:
            show[col] = show[col].round(2)
        show["IV"] = (show["IV"] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True)
        st.download_button(
            "Download trade log (CSV)", trades.to_csv(index=False),
            f"trade_log_{ticker}_{strat_name}.csv".replace(" ", "_"),
            "text/csv")

    st.caption(
        "Modeled with Black-Scholes (European options, "
        f"{'fixed ' + f'{fixed_iv:.0%}' + ' IV' if mode == 'fixed' else 'trailing 63-day realized vol'}, "
        f"r={rf:.1%}). Strikes set by delta target at entry; positions held "
        "to expiry, rolled immediately. No bid/ask spread, no commissions, "
        "no early exercise. Fixed size (1 contract / 100 shares), "
        "non-compounded. Hypothetical backtest — not investment advice.")

# ========================= OPTIONS ANALYTICS =========================
elif tool == "Options Analytics":
    st.title("Options Analytics")
    st.caption("Greeks and the volatility surface on the live CBOE delayed "
               "option chain (~15 min).")
    show_disclaimer()
    gl.show_glossary("Options Analytics")

    st.sidebar.header("Chain")
    oa_ticker = st.sidebar.text_input("Ticker", value="NVDA").strip().upper()

    if not oa_ticker:
        st.warning("Enter a ticker.")
        st.stop()

    with st.spinner(f"Fetching {oa_ticker} option chain…"):
        try:
            chain, spot, fetched_at = load_chain_bundle(oa_ticker)
            q = div_yield(oa_ticker)
            rf = risk_free_rate()
        except RuntimeError:
            st.error(f"Couldn't load an option chain for {oa_ticker} — "
                     "check the symbol (US stocks, ETFs and index options) "
                     "and try again in a minute.")
            st.stop()

    exp_list = [e for e in vol.expiries(chain) if vol.dte(e) >= 1]
    if not exp_list:
        st.error("No expiries in the chain — please try again in a minute.")
        st.stop()
    front = exp_list[0]
    ts = vol.term_structure(chain, spot)
    sk = vol.skew(chain, front, spot)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Underlying", f"${spot:,.2f}")
    c2.metric(f"ATM IV ({vol.dte(front)}d)",
              f"{sk['atm_iv']:.1%}" if sk["atm_iv"] else "—")
    rr = sk["risk_reversal_25"]
    c3.metric("25Δ risk reversal", f"{rr:+.1%}" if rr is not None else "—")
    slope = None
    atm_series = ts["atm_iv"].dropna()
    if len(atm_series) >= 2:
        slope = atm_series.iloc[-1] - atm_series.iloc[0]
    c4.metric("Term slope (ATM IV)", f"{slope:+.1%}"
              if slope is not None else "—")

    exp = st.selectbox("Expiry", exp_list,
                       format_func=lambda e: f"{e.date()} ({vol.dte(e)}d)")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Vol smile — {exp.date()}")
        sm = vol.smile(chain, exp)
        sfig = go.Figure()
        sfig.add_trace(go.Scatter(x=sm["strike"], y=sm["call_iv"] * 100,
                                  name="Calls", mode="lines+markers"))
        sfig.add_trace(go.Scatter(x=sm["strike"], y=sm["put_iv"] * 100,
                                  name="Puts", mode="lines+markers"))
        sfig.add_vline(x=spot, line_dash="dash", line_color="gray")
        sfig.update_layout(yaxis_title="IV %", xaxis_title="Strike",
                           height=340, margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(sfig, use_container_width=True)
    with right:
        st.subheader("Term structure — ATM IV")
        tfig = go.Figure()
        tfig.add_trace(go.Scatter(x=ts["dte"], y=ts["atm_iv"] * 100,
                                  mode="lines+markers", name="ATM IV"))
        tfig.update_layout(yaxis_title="IV %", xaxis_title="Days to expiry",
                           height=340, margin=dict(t=10, b=10, l=10, r=10),
                           showlegend=False)
        st.plotly_chart(tfig, use_container_width=True)

    st.subheader("Skew — 25Δ risk reversal & butterfly by expiry")
    kfig = go.Figure()
    kfig.add_trace(go.Scatter(x=ts["dte"], y=ts["risk_reversal_25"] * 100,
                              mode="lines+markers", name="Risk reversal 25Δ"))
    kfig.add_trace(go.Scatter(x=ts["dte"], y=ts["butterfly_25"] * 100,
                              mode="lines+markers", name="Butterfly 25Δ"))
    kfig.add_hline(y=0, line_dash="dash", line_color="gray")
    kfig.update_layout(yaxis_title="Vol points",
                       xaxis_title="Days to expiry", height=300,
                       margin=dict(t=10, b=10, l=10, r=10),
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(kfig, use_container_width=True)
    st.caption("Risk reversal < 0: downside puts pricier than upside calls — "
               "the usual equity put skew. Butterfly: smile convexity.")

    ivp = ivh.iv_percentile_row(oa_ticker)
    if ivp is not None:
        st.subheader("IV percentile — trailing history")
        i1, i2 = st.columns(2)
        i1.metric("Current ATM IV", f"{ivp['current_iv']:.1%}")
        i2.metric("IV percentile", f"{ivp['percentile']:.0f}")
        st.progress(min(max(ivp["percentile"] / 100.0, 0.0), 1.0),
                    text=f"{ivp['percentile']:.0f} of 100 — "
                         f"{'rich' if ivp['percentile'] >= 70 else 'cheap' if ivp['percentile'] <= 30 else 'mid-range'} vs history")
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=ivp["series"]["date"],
                                  y=ivp["series"]["atm_iv"] * 100,
                                  mode="lines+markers", name="ATM IV",
                                  line=dict(width=2.5)))
        pfig.update_layout(yaxis_title="IV %", height=260,
                           margin=dict(t=10, b=10, l=10, r=10),
                           showlegend=False)
        st.plotly_chart(pfig, use_container_width=True)
        st.caption(f"ATM IV vs its own history — {ivp['n']} daily chain "
                   f"snapshots, as of {ivp['as_of']}. 0 = cheapest on "
                   f"record, 100 = richest.")

    st.subheader("Gamma exposure (GEX)")
    st.caption("Dealer positioning: long calls / short puts. Positive GEX "
               "dampens moves (pinning); negative GEX amplifies them.")
    with st.spinner("Computing gamma exposure…"):
        try:
            gg = gex.gex_by_strike(chain, spot)
        except RuntimeError:
            gg = None
    if gg is None:
        st.warning("No usable open interest / gamma in the chain.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net GEX", f"${gg['total_net']:+.0f}M/pt")
        m2.metric("Put wall (support)",
                  f"{gg['put_wall']:.0f}" if gg["put_wall"] else "—")
        m3.metric("Call wall (resistance)",
                  f"{gg['call_wall']:.0f}" if gg["call_wall"] else "—")
        m4.metric("γflip",
                  f"{gg['zero_gamma']:.0f}" if gg["zero_gamma"] else "—")
        gfig = gex.gex_chart(gg, f"Net GEX by strike — {oa_ticker} ($M/pt)")
        st.plotly_chart(gfig, use_container_width=True)
        exp_str = ", ".join(pd.Timestamp(e).strftime("%b %d")
                            for e in gg["expiries"])
        st.caption(gex.gex_read(gg) + f" Nearest 3 expiries: {exp_str}.")

    gh = ivh.gex_history_rows(oa_ticker)
    if gh is not None:
        st.subheader("GEX history")
        hfig = go.Figure()
        hfig.add_trace(go.Scatter(x=gh["gex"]["date"],
                                  y=gh["gex"]["net_gex_m"],
                                  name="Net GEX ($M/pt)",
                                  line=dict(width=2.5)))
        hfig.add_hline(y=0, line_dash="dash", line_color="gray")
        if not gh["flip"].empty:
            hfig.add_trace(go.Scatter(x=gh["flip"]["date"],
                                      y=gh["flip"]["gamma_flip"],
                                      name="γflip", yaxis="y2",
                                      line=dict(width=2, dash="dot",
                                                color="#f1c40f")))
            hfig.update_layout(yaxis2=dict(title="γflip strike",
                                           overlaying="y", side="right",
                                           showgrid=False))
        hfig.update_layout(yaxis_title="Net GEX ($M/pt)", height=300,
                           margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(hfig, use_container_width=True)
        st.caption(f"Daily chain snapshots, as of {gh['as_of']}. Net GEX "
                   f"from the nearest 3 expiries each day.")

    st.subheader("Greeks by strike")
    gname = st.selectbox("Greek", ["Delta", "Gamma", "Theta", "Vanna", "Charm"],
                         index=1)
    gkey = gname.lower()
    glabel = {"delta": "Δ (per share)", "gamma": "Γ (per $1 move)",
              "theta": "Θ ($/day)", "vanna": "dΔ per vol point",
              "charm": "dΔ per day"}[gkey]
    with st.spinner("Computing greeks…"):
        cg = vol.chain_greeks(chain, exp, spot, rf, q)
    if cg.empty:
        st.warning("No quoted contracts for this expiry.")
    else:
        gfig = go.Figure()
        for otype, nm in (("call", "Calls"), ("put", "Puts")):
            sub = cg[cg["type"] == otype]
            gfig.add_trace(go.Scatter(x=sub["strike"], y=sub[gkey],
                                      mode="lines+markers", name=nm))
        gfig.update_layout(yaxis_title=glabel, xaxis_title="Strike",
                           height=340, margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(gfig, use_container_width=True)
        if gkey == "vanna":
            st.caption("Vanna > 0: the position's delta grows if vol rises — "
                       "long-vol positions get longer into a spike.")
        elif gkey == "charm":
            st.caption("Charm: delta bleed per passing day, all else equal.")

        st.subheader(f"Chain — {exp.date()} (model greeks)")
        wide = cg.pivot(index="strike", columns="type")
        wide.columns = [f"{c}_{t}" for c, t in wide.columns]
        disp = pd.DataFrame(index=wide.index)
        for t in ("call", "put"):
            T = t[0].upper()
            if f"iv_{t}" in wide:
                disp[f"{T} IV %"] = (wide[f"iv_{t}"] * 100).round(1)
            for g_, dec in (("delta", 3), ("gamma", 4), ("theta", 4),
                            ("vanna", 4), ("charm", 4)):
                col = f"{g_}_{t}"
                if col in wide:
                    disp[f"{T} {g_}"] = wide[col].round(dec)
            if f"open_interest_{t}" in wide:
                disp[f"{T} OI"] = wide[f"open_interest_{t}"].fillna(0).astype(int)
            if f"volume_{t}" in wide:
                disp[f"{T} vol"] = wide[f"volume_{t}"].fillna(0).astype(int)
        disp.index.name = "Strike"
        st.dataframe(disp, use_container_width=True)

    if oa_ticker == "VIX":
        exercise_note = ("VIX options are on futures — greeks approximate.")
    elif oa_ticker in ("SPX", "RUT", "NDX"):
        exercise_note = ("Index options are European-style (cash-settled).")
    else:
        exercise_note = ("Listed equity options are American-style, so "
                         "deep-ITM greeks are approximate.")
    st.caption(f"Chain as of {fetched_at:%H:%M} UTC (CBOE delayed ~15 min) · "
               f"r={rf:.1%} (3M T-bill), q={q:.2%} · Greeks via Black-Scholes "
               f"(European); {exercise_note} Not investment advice.")


# ============================== CHANGELOG ==============================
with st.sidebar.expander("Changelog"):
    st.markdown(
        (Path(__file__).resolve().parent / "CHANGELOG.md").read_text())
