# Portfolio Lab

Three tools in one Streamlit app:

**Asset Mix Dashboard** — define a hypothetical asset mix (tickers + target
weights) and see what it would have done: allocation, growth of capital vs
benchmark, risk stats (CAGR, volatility, Sharpe, max drawdown), drawdowns,
holding correlations, and weight drift.

**Strategy Tester** — backtest stock/option strategies on historical prices
with options priced by Black-Scholes (European, no early exercise, no
bid/ask). Strategies: Buy & Hold, Covered Call, Cash-Secured Put, Bull Call
Spread, Bear Put Spread, Long Straddle, Iron Condor. Strikes are set by delta
target at entry; positions are held to expiry and rolled immediately.

**Options Analytics** — greeks and the volatility surface on the live CBOE
delayed option chain (~15 min, no key) for any US stock, ETF, or index
option — type any ticker (SPX/RUT/NDX/VIX map to CBOE's `_`-prefixed
index symbols):
- Greeks: delta, gamma, theta, vanna, charm per contract (`greeks.py`,
  Black-Scholes with dividend yield; theta/charm per calendar day, vanna per
  vol point). Vanna and charm are not published by CBOE — computed here.
- Volatility: smile (IV vs strike per expiry), 25-delta risk reversal and
  butterfly skew across expiries, ATM-IV term structure (`volsurface.py`,
  using CBOE's listed IVs).
- Gamma exposure: net GEX by strike ($M/pt) over the nearest 3 expiries,
  bars stacked/colored by expiry, with call/put walls and γflip (`gex.py`)
  — same convention as the US Market Sentiment app.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

New app → existing repo `louisctlau/asset-mix-dashboard`, branch `main`,
main file `app.py`.

## Notes

- Data: Yahoo Finance adjusted closes (dividends reinvested), cached 24h.
- Rebalancing: monthly / quarterly / annual / none.
- Benchmarks: SPY or a monthly-rebalanced 60/40 SPY/BND.
- Options Analytics: chain cached 15 min; expiries under 1 day out are
  excluded (0DTE smiles are degenerate). Greeks verified two ways — 80
  analytic-vs-finite-difference checks (delta, gamma, theta, vega, vanna,
  charm, IV round-trip, call/put parity) and cross-check vs CBOE's listed
  delta/gamma/theta on the live chain (100% sign agreement; magnitude
  converges to ~1.0 with expiry; CBOE floors tiny deep-ITM gamma to zero).
  Listed equity options are American-style, so deep-ITM model greeks are
  approximate.
- Hypothetical backtest — not investment advice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

New app → existing repo `louisctlau/asset-mix-dashboard`, branch `main`,
main file `app.py`.

## Notes

- Data: Yahoo Finance adjusted closes (dividends reinvested), cached 24h.
- Rebalancing: monthly / quarterly / annual / none.
- Benchmarks: SPY or a monthly-rebalanced 60/40 SPY/BND.
- Hypothetical backtest — not investment advice.
