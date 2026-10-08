# Changelog — Portfolio Lab

## 2026-10-08 — GEX chart: legend fix + stat row
- The GEX chart legend moved to the top-right inside the plot (with a dark
  backing) — it no longer overlaps the chart title or the x-axis label.
- New stat row under the Gamma Exposure title, matching the US Market
  Sentiment app: Net GEX ($M/pt), Put wall (support), Call wall
  (resistance), γflip.

## 2026-10-05 — Options Analytics: index options
- Index tickers now work: SPX, RUT, NDX and VIX map to CBOE's
  underscore-prefixed symbols (`_SPX` …). Index options are European-style
  (cash-settled), so the Black-Scholes greeks are exact; q=0 as Yahoo
  publishes no index dividend yield. VIX options are on futures — greeks
  approximate.

## 2026-10-05 — GEX chart: expiry color-coding
- The GEX chart now stacks each strike's bar by expiry (one color per
  expiry, positive gamma right / negative left, horizontal) — the bar total
  is still the net per strike. Spot, walls and γflip overlaid as before.

## 2026-10-05 — Options Analytics: auto risk-free rate
- The manual risk-free rate input is gone — the tab now uses the live 3M
  T-bill yield (^IRX via Yahoo, 24h cache, 4% fallback), shown in the
  footer (e.g. "r=4.0% (3M T-bill)").

## 2026-10-05 — Gamma exposure (GEX)
- New GEX section in Options Analytics: net gamma exposure by strike
  ($M/pt) over the nearest 3 expiries, with call/put walls and the
  zero-gamma level — same formula and dealer-positioning convention as the
  US Market Sentiment app (`GEX = (call OI × call γ − put OI × put γ) × 100
  × spot`, CBOE listed gamma). Cross-validated bit-identical vs USMS on
  SPY (net +$754M, walls 776/770, 0γ 772).
- The zero-gamma line is labeled "γflip" (chart + read line).

## 2026-10-05 — Options Analytics: any-ticker input
- The Ticker control is now a free-text input — any US stock or ETF with
  listed options works (verified live: TSLA, QQQ, SPY). Cash index options
  (SPX, RUT, NDX, VIX) are blocked by CBOE on the free delayed endpoint;
  SPY/QQQ serve as the liquid index proxies.

## 2026-10-05 — Options Analytics tab
- New third tool: greeks (delta, gamma, theta, vanna, charm) and the
  volatility surface — smile (IV vs strike per expiry), 25Δ risk reversal
  and butterfly skew across expiries, ATM-IV term structure — on the live
  CBOE delayed option chain (~15 min, no key). New modules `greeks.py`
  (Black-Scholes, verified against finite differences), `chains.py` (chain
  fetch), `volsurface.py` (smile/skew/term structure).
- Dividend yield now uses Yahoo's `trailingAnnualDividendYield` (a true
  ratio). The old `dividendYield` field is in percent units and overstated
  q ~100× (showed q=43.00% for NVDA).

## 2026-09-28 — Strategy Tester tab
- Backtest 7 stock/option strategies (Buy & Hold, Covered Call,
  Cash-Secured Put, Bull Call Spread, Bear Put Spread, Long Straddle, Iron
  Condor) on historical prices with Black-Scholes pricing (European, no
  early exercise, no bid/ask). Strikes set by delta target at entry,
  positions held to expiry and rolled immediately.

## 2026-09-28 — App launch: Asset Mix Dashboard
- Hypothetical portfolio builder: editable ticker/weight table, allocation
  donut, growth of capital vs SPY or 60/40 benchmark, CAGR, volatility,
  Sharpe, max drawdown, best/worst year, drawdown chart, holding
  correlations, weight drift vs target.
- Fixed the weight-drift column rendering 100× too large.
