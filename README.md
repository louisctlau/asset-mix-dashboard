# Portfolio Lab

Two tools in one Streamlit app:

**Asset Mix Dashboard** — define a hypothetical asset mix (tickers + target
weights) and see what it would have done: allocation, growth of capital vs
benchmark, risk stats (CAGR, volatility, Sharpe, max drawdown), drawdowns,
holding correlations, and weight drift.

**Strategy Tester** — backtest stock/option strategies on historical prices
with options priced by Black-Scholes (European, no early exercise, no
bid/ask). Strategies: Buy & Hold, Covered Call, Cash-Secured Put, Bull Call
Spread, Bear Put Spread, Long Straddle, Iron Condor. Strikes are set by delta
target at entry; positions are held to expiry and rolled immediately.

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
