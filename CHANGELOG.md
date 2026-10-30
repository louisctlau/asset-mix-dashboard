# Changelog — Portfolio Lab

## 2026-10-30 — 5% rebalance bands, Strategy Tester costs, put/call columns
- Asset Mix — new "5% bands" rebalance mode: holdings reset to target
  weights at the next open after any holding drifts more than 5 percentage
  points from target (calendar Monthly/Quarterly/Annual/None unchanged;
  the rebalancing check, weight-drift table, share links and stress replay
  all accept the new mode).
- Strategy Tester — commissions and slippage: sidebar inputs ($0.65 /
  contract / side commission, $0.01 / share / side slippage by default,
  charged on open and close; stock legs carry no commission). Each trade
  now shows Costs and P&L (net), the metrics add Total P&L (net) and
  Return on capital (net), and the equity curve adds a net-of-costs line.
  The side-by-side comparison stays gross of costs, stated in its caption.
- Data — options_history.csv gains put_oi / call_oi / pc_oi columns
  (total put/call open interest and their ratio per ticker per day, from
  the archive builder), feeding the USMS put/call panel; the builder
  script is now part of the push file list so it ships with the repo.

## 2026-10-25 — Changelog as its own page
- Changelog moved out of the sidebar expander into its own page
  (`pages/changelog.py`, USMS pattern), linked from the bottom of the
  sidebar, with a back link to the dashboard. Auto sidebar navigation
  is hidden so the link is the only route to it.

## 2026-10-22 — Compare mixes, look-through, stress replay, live-IV mode, share links
- Asset Mix: save the current mix to slots A/B/C and compare them in a new
  "Compare saved mixes" section — overlaid growth curves on the same
  lookback/capital/rebalance/benchmark, with a metrics table (total
  return, CAGR, volatility, Sharpe, max drawdown, best/worst year) and
  the best value in each row highlighted.
- Asset Mix: look-through exposure (new module `lookthrough.py`) — each
  fund's disclosed top holdings aggregated through the portfolio weights
  into a top-underlyings table and sector breakdown, with an explicit
  coverage line (share of portfolio weight with holdings data; any
  tickers without data are named, and the beyond-top-10 remainder is
  reported as unattributed rather than guessed).
- Asset Mix: rebalance counterfactual — when a rebalance schedule is
  selected, a callout reports the same mix never rebalanced (CAGR and
  total return deltas), computed on the existing backtest path.
- Asset Mix: stress replay (new module `stress.py`) — the current mix
  through three fixed windows (COVID crash Feb–Mar 2020 with days to
  recover the pre-crash peak, the 2022 bear market, the 2018 Q4 selloff)
  on full price history. A holding whose history starts after a window
  opens is excluded from that episode and named in a footnote; weights
  renormalize over the rest, never silently.
- Asset Mix: share links — the mix (tickers, weights, capital, lookback,
  rebalance, benchmark) is encoded in the page URL query parameters and
  shown as a copyable share link; opening a shared link prefills the
  builder. New download: metrics + weights as CSV.
- Strategy Tester: third volatility source, "Live ATM IV (CBOE chain)" —
  today's front-expiry ATM IV from the live delayed chain, applied as the
  fixed vol for the whole run and labelled as such (it is today's IV,
  not a historical IV series). If the chain can't be fetched, the tester
  falls back to trailing 63d realized vol with a warning.
- Options Analytics: the IV percentile now ranks today's LIVE ATM IV
  against the archived daily series, and the panel labels the series
  honestly — accrual start date, number of sessions, archive coverage
  (the six archived names). The term-structure chart marks the expiry
  bracketing the next earnings date (Yahoo Finance calendar, labelled
  as possibly an estimate) where one is known.

## 2026-10-18 — Strategy Tester: blank CAGR now reads "n/a" + footnote
- In the side-by-side comparison table, a strategy whose cumulative losses
  exceed its first-trade capital (equity curve crosses zero) made the CAGR
  cell blank. It now reads "n/a", with a caption explaining that CAGR is
  undefined in that case.

## 2026-10-10 — History panels, rolling metrics, Monte Carlo, presets, glossary
- New daily options-history builder (`scripts/build_options_history.py`):
  reads the options-chain archive and writes `data/options_history.csv`
  (date, ticker, net GEX, γflip, call/put walls, ATM IV, spot).
- Options Analytics: IV percentile panel — current ATM IV ranked against its
  own trailing history, with a history line chart (hides silently when the
  ticker has fewer than 5 history points).
- Options Analytics: GEX history chart — trailing net GEX and γflip from the
  daily archive.
- Asset Mix: rolling 1-year Sharpe, volatility and max drawdown charts
  (portfolio + faint benchmark).
- Asset Mix: Monte Carlo fan chart — 2,000 simulated 1-year paths from the
  portfolio's own monthly returns, with 10th/50th/90th percentile bands
  (buy-and-hold assumed within paths; rebalance not modeled).
- Strategy Tester: side-by-side comparison table running all 7 strategies on
  the same inputs (total return, CAGR, Sharpe, max DD, win rate, # trades).
- Asset Mix sidebar: one-click presets (60/40, All Equity, Conservative,
  Golden Butterfly, Permanent Portfolio) plus JSON download/upload for
  custom mixes.
- "What am I looking at?" glossary expanders in all three tools.
- CSV downloads: growth series, strategy comparison table, trade log.

## 2026-10-08 — GEX chart: legend fix + stat row
- The GEX chart legend moved to the top-right inside the plot (with a dark
  backing) — it no longer overlaps the chart title or the x-axis label.
- New stat row under the Gamma Exposure title, matching the US Market
  Sentiment app: Net GEX ($M/pt), Put wall (support), Call wall
  (resistance), γflip.
- Thin grey gridlines on the GEX chart for easier reading.
- Disclaimer under each tool's title: "For educational and experimental
  purposes only. Not investment advice." (red).

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
