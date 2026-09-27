"""Volatility-forecasting baselines that TDA features have to beat."""

import numpy as np


def trailing_rv(x, ends, k):
    """Realized variance over the ``k`` days ending at each index in ``ends`` (inclusive)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=float) ** 2)])
    ends = np.asarray(ends)
    return (c[ends + 1] - c[ends + 1 - k]) / k


def future_rv(x, ends, horizon):
    """Realized variance over the ``horizon`` days after each index in ``ends``: the target."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=float) ** 2)])
    ends = np.asarray(ends)
    return (c[ends + 1 + horizon] - c[ends + 1]) / horizon


def har_design(x, ends, lags=(5, 22, 100)):
    """HAR-style regressors from daily data: log trailing realized variances at several horizons."""
    return np.column_stack([np.log(trailing_rv(x, ends, k)) for k in lags])


def garch_filter(x, omega, alpha, beta):
    """Conditional variances of a GARCH(1,1) run over ``x``: ``h[t]`` is the variance of ``x[t]``.

    ``h`` has length ``len(x) + 1``. The last entry is the one-step-ahead forecast.
    """
    persistence = alpha + beta
    h_prev = omega / (1.0 - persistence) if persistence < 1.0 else float(np.var(x))
    h = np.empty(len(x) + 1)
    h[0] = h_prev
    for t, x_t in enumerate(np.asarray(x, dtype=float).tolist()):
        h_prev = omega + alpha * x_t * x_t + beta * h_prev
        h[t + 1] = h_prev
    return h


def garch_mean_forecast(h_next, omega, alpha, beta, horizon):
    """E[mean of h over the next ``horizon`` days | today], given tomorrow's variance ``h_next``.

    Uses E[h_{t+j}] = hbar + (alpha + beta)^(j-1) (h_{t+1} - hbar).
    """
    p = alpha + beta
    hbar = omega / (1.0 - p)
    decay = np.mean(p ** np.arange(horizon))
    return hbar + decay * (np.asarray(h_next) - hbar)
