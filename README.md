# Asset Mix Dashboard

A Streamlit app for analyzing a **hypothetical portfolio**: define an asset
mix (tickers + target weights) and see what it would have done — allocation,
growth of capital vs benchmark, risk stats (CAGR, volatility, Sharpe, max
drawdown), drawdowns, holding correlations, and weight drift.

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
