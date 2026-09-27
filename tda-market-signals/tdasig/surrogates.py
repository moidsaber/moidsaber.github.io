"""Surrogate series. Each null keeps some structure of the data and destroys the rest.

shuffle  permute days within each window. Keeps each window's marginal (and its
         vol) and kills all temporal order.
fourier  randomize Fourier phases. Keeps the periodogram, hence the autocovariance,
         and makes the series Gaussian otherwise. This is the linear-Gaussian null.
iaaft    iterated amplitude-adjusted Fourier surrogate. Keeps the exact marginal
         and (almost exactly) the periodogram: the null "a linear Gaussian process
         seen through a monotone map". It kills nonlinear temporal structure such
         as vol clustering or a stable oscillation amplitude.
garch    simulate a GARCH(1,1)-t fitted to the data. Keeps vol clustering and fat
         tails and kills everything else.
"""

import warnings

import numpy as np

from .simulate import garch


def shuffle_windows(W, rng):
    """Independently permute the days inside every row of the window matrix ``W``."""
    return rng.permuted(W, axis=1)


def phase_randomize(x, rng):
    """Fourier surrogate: same amplitude spectrum as ``x``, independent uniform phases."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    spec = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0.0, 2.0 * np.pi, len(spec))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0  # the Nyquist coefficient of a real series is real
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n) + x.mean()


def iaaft(x, rng, n_iter=200):
    """IAAFT surrogate (Schreiber & Schmitz 1996): exact marginal, near-exact periodogram.

    Alternates between imposing the target Fourier amplitudes and rank-remapping
    onto the sorted data values. It ends on a remap step, so the marginal is exact.
    """
    x = np.asarray(x, dtype=float)
    amp = np.abs(np.fft.rfft(x))
    values = np.sort(x)
    s = rng.permutation(x)
    for _ in range(n_iter):
        s = np.fft.irfft(amp * np.exp(1j * np.angle(np.fft.rfft(s))), len(x))
        s = values[np.argsort(np.argsort(s))]
    return s


def fit_garch_t(x):
    """Maximum-likelihood zero-mean GARCH(1,1) with standardized Student-t innovations."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="t",
                         rescale=False).fit(disp="off")
    p = res.params
    return {"omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
            "beta": float(p["beta[1]"]), "nu": float(p["nu"])}


def garch_t_surrogate(params, n, rng, burn=1000):
    """One series of length ``n`` simulated from fitted ``fit_garch_t`` parameters."""
    r, _ = garch(n, params["alpha"], params["beta"], rng, nu=params["nu"],
                 omega=params["omega"], burn=burn)
    return r
