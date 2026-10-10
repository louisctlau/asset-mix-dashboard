"""Glossary panels ("What am I looking at?") for the three Portfolio Lab tools.

Plain-language definitions aimed at a non-specialist reader — e.g. a hiring
manager opening a portfolio link. Rendered via show_glossary(tool).
"""
from __future__ import annotations

import streamlit as st

TERMS: dict[str, dict[str, str]] = {
    "Asset Mix": {
        "CAGR": "Compound annual growth rate — the steady yearly rate that "
                "would take the starting value to the ending value.",
        "Volatility (ann.)": "Annualized standard deviation of daily "
                "returns — how much the portfolio's value typically wobbles "
                "in a year. Higher = bumpier ride.",
        "Sharpe ratio": "Return per unit of risk taken. Above 1 is good, "
                "above 2 is excellent; here computed with a 0% risk-free "
                "rate for comparability.",
        "Max drawdown": "The worst peak-to-trough fall in the period — the "
                "most you would have lost buying at the top and selling at "
                "the bottom.",
        "Weight drift": "How far each holding has drifted from its target "
                "weight as winners outgrow losers. Rebalancing resets it.",
        "Rebalancing": "Periodically selling winners and buying losers to "
                "restore target weights — a disciplined 'sell high, buy "
                "low'.",
        "Rolling metrics": "The same stat recomputed on a trailing 1-year "
                "window, rolled forward day by day — shows whether risk and "
                "return are stable or regime-shifting.",
        "Monte Carlo fan chart": "2,000 simulated 1-year futures built by "
                "reshuffling the portfolio's own historical monthly "
                "returns. The bands show the 10th–90th percentile range of "
                "outcomes — a plan, not a prediction.",
    },
    "Strategy Tester": {
        "Black-Scholes": "The standard formula for pricing European "
                "options. This tester prices every option with it — real "
                "market prices are not used.",
        "Implied volatility (IV)": "The market's forecast of future "
                "wiggle, baked into option prices. Higher IV = pricier "
                "options.",
        "Delta": "How much an option's price moves per $1 move in the "
                "stock. Used here to pick strikes (e.g. 30-delta call).",
        "Covered call": "Own 100 shares, sell a call against them — "
                "collects premium, caps upside.",
        "Cash-secured put": "Sell a put while holding the cash to buy the "
                "shares if assigned — collects premium, obliges a purchase "
                "below the strike.",
        "Bull call / bear put spread": "Buy one option, sell a further "
                "out-of-the-money one — cheaper directional bet with "
                "capped profit and loss.",
        "Long straddle": "Buy a call and a put at the same strike — "
                "profits if the stock moves a lot either way.",
        "Iron condor": "Sell an out-of-the-money call spread and put "
                "spread — profits if the stock stays in a range.",
        "Win rate": "Share of trades that finished profitable.",
        "Profit factor": "Gross profit divided by gross loss. Above 1 "
                "means the strategy made money overall.",
    },
    "Options Analytics": {
        "GEX (gamma exposure)": "How much dealer hedging flow a $1 move in "
                "the stock triggers, in $M per point. Positive GEX: dealers "
                "are long gamma and their hedging dampens moves (pinning). "
                "Negative GEX: dealers are short gamma and hedging "
                "amplifies moves.",
        "γflip (gamma flip)": "The strike where net dealer gamma changes "
                "sign. Above it, moves are dampened; below it, they can "
                "accelerate.",
        "Put wall / call wall": "Strikes with the heaviest put/call gamma "
                "positioning — often act as support (put wall) and "
                "resistance (call wall).",
        "IV percentile": "Where today's at-the-money IV ranks against its "
                "own history (0 = cheapest ever, 100 = richest ever). High "
                "percentile favors option sellers; low favors buyers.",
        "Vol smile": "IV plotted against strike — usually U-shaped for "
                "equities because downside puts trade rich.",
        "Skew / risk reversal": "25-delta risk reversal = IV of 25-delta "
                "call minus 25-delta put. Negative = downside protection "
                "costs more than upside (normal for equities).",
        "Butterfly": "Measures smile convexity — how much wings trade "
                "rich to at-the-money.",
        "Term structure": "ATM IV across expiries. Upward-sloping "
                "(contango) is normal; inversion can signal near-term "
                "stress.",
        "Vanna": "How delta changes when IV moves — the vol-directional "
                "Greek.",
        "Charm": "How delta decays with passing time — delta bleed per "
                "day.",
        "Open interest": "Outstanding contracts — prior-day positioning, "
                "not today's trading.",
    },
}


def show_glossary(tool: str) -> None:
    terms = TERMS.get(tool)
    if not terms:
        return
    with st.expander("What am I looking at? — glossary"):
        for term, definition in terms.items():
            st.markdown(f"**{term}** — {definition}")
