"""Asset Mix Dashboard — hypothetical portfolio analysis.

Define a hypothetical asset mix in the sidebar; the app backtests it on
Yahoo Finance total-return data (dividends reinvested) and reports allocation,
growth, risk stats, drawdowns, correlations, and weight drift.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Asset Mix Dashboard", layout="wide")
st.title("Asset Mix Dashboard")
st.caption("Hypothetical portfolio — what would this asset mix have done?")

PERIODS = {"1Y": "1y", "3Y": "3y", "5Y": "5y", "10Y": "10y", "Max": "max"}
TRADING_DAYS = 252

DEFAULT_HOLDINGS = pd.DataFrame(
    {"Ticker": ["SPY", "QQQ", "BND", "GLD"], "Weight %": [50.0, 10.0, 30.0, 10.0]}
)


@st.cache_data(ttl=86400, show_spinner=False)
def load_closes(tickers: tuple, period: str) -> pd.DataFrame:
    """Adjusted closes (dividends reinvested) for the given tickers."""
    data = yf.download(list(tickers), period=period, auto_adjust=True,
                       progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:  # single ticker
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})
    return closes.dropna(how="all")


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


def stats(growth: pd.Series, port_rets: pd.Series) -> dict:
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


# ---------------- Sidebar: portfolio builder ----------------
st.sidebar.header("Hypothetical portfolio")
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
                                  value=10000, step=1000)
lookback = st.sidebar.selectbox("Lookback", list(PERIODS), index=2)
rebalance = st.sidebar.selectbox(
    "Rebalance", ["Monthly", "Quarterly", "Annual", "None"], index=1)
benchmark = st.sidebar.selectbox("Benchmark", ["SPY", "60/40 (SPY/BND)", "None"],
                                 index=0)

# ---------------- Validate & load ----------------
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
    st.sidebar.caption(f"Weights sum to {weights.sum():.1f}% — normalized to 100%.")
weights = weights / weights.sum()

needed = set(tickers)
if benchmark == "60/40 (SPY/BND)":
    needed |= {"SPY", "BND"}
elif benchmark == "SPY":
    needed.add("SPY")

with st.spinner("Fetching market data…"):
    closes = load_closes(tuple(sorted(needed)), PERIODS[lookback])

bad = [t for t in needed if t not in closes.columns
       or closes[t].dropna().empty]
if bad:
    st.warning(f"Couldn't fetch data for: {', '.join(sorted(bad))} — excluded.")
for t in bad:
    needed.discard(t)
    if t in weights.index:
        weights = weights.drop(t)
if weights.empty:
    st.error("No usable holdings left.")
    st.stop()
weights = weights / weights.sum()
closes = closes.dropna()

prices = closes[[t for t in weights.index]]
port_rets, whist = simulate(prices, weights, rebalance)
growth = (1 + port_rets).cumprod() * capital

# ---------------- Benchmarks ----------------
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

# ---------------- Layout ----------------
s = stats(growth, port_rets)
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
        "Target": weights,
        "Current": whist.iloc[-1],
    })
    drift["Drift (pp)"] = (drift["Current"] - drift["Target"]) * 100
    drift = (drift * 100).round(1)
    drift.columns = ["Target %", "Current %", "Drift (pp)"]
    st.dataframe(drift.sort_values("Drift (pp)", key=abs, ascending=False),
                 use_container_width=True)
    st.caption(f"Rebalancing: {rebalance.lower()} · "
               f"{growth.index[0].date()} → {growth.index[-1].date()}")

st.caption("Data: Yahoo Finance adjusted closes (dividends reinvested), "
           "daily. Hypothetical backtest — not investment advice.")
