"""Black-Scholes greeks engine: delta, gamma, theta, vega, vanna, charm.

European exercise, continuous dividend yield q. Desk conventions:
- theta: $ per share per calendar day (value change as one day passes)
- vega:  $ per share per 1 vol point (0.01 of sigma)
- vanna: change in delta per 1 vol point move in implied vol
- charm: change in delta per calendar day passing (all else equal)

Analytic formulas are cross-checked against finite differences in the
verification script (see README); charm's sign convention is pinned to
"delta tomorrow minus delta today".
"""
from math import erf, exp, log, pi, sqrt

_DAYS = 365.0
_TWO_PI = 2.0 * pi


def _pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(_TWO_PI)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _d1_d2(S: float, K: float, T: float, r: float, q: float,
           sig: float) -> tuple[float, float]:
    sq = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sq)
    return d1, d1 - sig * sq


def price(S: float, K: float, T: float, r: float, q: float, sig: float,
          kind: str) -> float:
    """Black-Scholes price for a European call/put."""
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    sig = max(sig, 1e-6)
    d1, d2 = _d1_d2(S, K, T, r, q, sig)
    if kind == "call":
        return S * exp(-q * T) * _cdf(d1) - K * exp(-r * T) * _cdf(d2)
    return K * exp(-r * T) * _cdf(-d2) - S * exp(-q * T) * _cdf(-d1)


def greeks(S: float, K: float, T: float, r: float, q: float, sig: float,
           kind: str) -> dict:
    """All six greeks (+price) for one European call/put.

    T in years; may be 0 (expiry) — greeks fall back to intrinsic values.
    """
    out = {"price": price(S, K, T, r, q, sig, kind)}
    if T <= 0 or S <= 0 or K <= 0:
        if kind == "call":
            out.update(delta=1.0 if S > K else 0.0, gamma=0.0, theta=0.0,
                       vega=0.0, vanna=0.0, charm=0.0)
        else:
            out.update(delta=-1.0 if S < K else 0.0, gamma=0.0, theta=0.0,
                       vega=0.0, vanna=0.0, charm=0.0)
        return out

    sig = max(sig, 1e-6)
    sq = sqrt(T)
    d1, d2 = _d1_d2(S, K, T, r, q, sig)
    nd1 = _pdf(d1)
    disc_q = exp(-q * T)
    disc_r = exp(-r * T)
    Nd1 = _cdf(d1)

    # delta
    delta = disc_q * Nd1 if kind == "call" else disc_q * (Nd1 - 1.0)

    # gamma (same for calls and puts)
    gamma = disc_q * nd1 / (S * sig * sq)

    # theta: value change per calendar day as time passes (<= 0 for longs)
    carry = S * disc_q * nd1 * sig / (2.0 * sq)
    if kind == "call":
        theta = (-carry - r * K * disc_r * _cdf(d2)
                 + q * S * disc_q * Nd1) / _DAYS
    else:
        theta = (-carry + r * K * disc_r * _cdf(-d2)
                 - q * S * disc_q * _cdf(-d1)) / _DAYS

    # vega: $ per 1 vol point
    vega = S * disc_q * nd1 * sq / 100.0

    # vanna: dDelta/dSigma, per 1 vol point (same for calls and puts)
    vanna = -disc_q * nd1 * d2 / sig / 100.0

    # charm: dDelta per calendar day passing, all else equal
    # dDelta/dT pinned by finite differences (see verification notes).
    d1_dT = (2.0 * (r - q) * T - d2 * sig * sq) / (2.0 * T * sig * sq)
    if kind == "call":
        charm = disc_q * (q * Nd1 - nd1 * d1_dT) / _DAYS
    else:
        charm = disc_q * (q * (Nd1 - 1.0) - nd1 * d1_dT) / _DAYS

    out.update(delta=delta, gamma=gamma, theta=theta, vega=vega,
               vanna=vanna, charm=charm)
    return out


def implied_vol(px: float, S: float, K: float, T: float, r: float, q: float,
                kind: str) -> float | None:
    """Implied vol by bisection; None if the price is not bracketable."""
    if T <= 0 or px <= 0:
        return None
    lo, hi = 1e-4, 5.0
    p_lo = price(S, K, T, r, q, lo, kind)
    p_hi = price(S, K, T, r, q, hi, kind)
    if not (p_lo <= px <= p_hi):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if price(S, K, T, r, q, mid, kind) < px:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
