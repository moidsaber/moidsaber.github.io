"""Return generators with known ground truth.

Each generator isolates one mechanism. All randomness comes from an explicit
``numpy.random.Generator``, so every experiment is reproducible from its seed.
"""

import numpy as np


def standardize(r):
    """Rescale a whole series to zero mean and unit variance.

    This is one global affine map, so every within-series property (clustering,
    tails, cycles, regimes) is untouched. It only puts all generators on the same
    unconditional scale, which makes their *raw* features comparable.
    """
    r = np.asarray(r, dtype=float)
    return (r - r.mean()) / r.std()


def gaussian(n, rng):
    """i.i.d. N(0, 1): the featureless-blob null."""
    return rng.standard_normal(n)


def student_t(n, nu, rng):
    """i.i.d. Student-t with ``nu`` degrees of freedom, scaled to unit variance (nu > 2)."""
    return rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)


def garch(n, alpha, beta, rng, nu=None, omega=None, burn=1000):
    """GARCH(1,1): r_t = sqrt(h_t) z_t,  h_t = omega + alpha r_{t-1}^2 + beta h_{t-1}.

    ``z_t`` is N(0, 1) when ``nu`` is None, otherwise unit-variance Student-t(nu).
    ``omega`` defaults to 1 - alpha - beta (unit unconditional variance).
    Returns ``(r, h)`` where ``h[t]`` is the conditional variance of ``r[t]``.
    """
    if omega is None:
        omega = 1.0 - alpha - beta
    total = n + burn
    z = rng.standard_normal(total) if nu is None else student_t(total, nu, rng)
    r = np.empty(total)
    h = np.empty(total)
    persistence = alpha + beta
    h_prev = omega / (1.0 - persistence) if persistence < 1.0 else omega / 1e-2
    r_prev = 0.0
    for t, z_t in enumerate(z.tolist()):
        h_t = omega + alpha * r_prev * r_prev + beta * h_prev
        r_t = h_t ** 0.5 * z_t
        h[t] = h_t
        r[t] = r_t
        h_prev, r_prev = h_t, r_t
    return r[burn:], h[burn:]


def markov_regimes(n, durations, rng):
    """Two-state Markov chain with expected sojourn times ``durations = (L0, L1)``.

    The initial state is drawn from the stationary distribution.
    """
    leave = 1.0 / np.asarray(durations, dtype=float)
    state = int(rng.random() < leave[0] / leave.sum())
    u = rng.random(n)
    s = np.empty(n, dtype=int)
    for t in range(n):
        s[t] = state
        if u[t] < leave[state]:
            state = 1 - state
    return s


def regime_garch(n, alpha, beta, scales, durations, rng, nu=None, burn=1000):
    """Regime-switching GARCH: r_t = scales[s_t] * u_t.

    ``u_t`` is a unit-variance GARCH(1,1) and ``s_t`` a two-state Markov chain
    (see ``markov_regimes``). Because the regime multiplies the scale, every
    switch is a sharp change point in volatility, while within a regime the
    dynamics are ordinary GARCH clustering. Returns ``(r, s)``.
    """
    u, _ = garch(n, alpha, beta, rng, nu=nu, burn=burn)
    s = markov_regimes(n, durations, rng)
    return np.asarray(scales, dtype=float)[s] * u, s


def switch_dates(states):
    """Indices ``t`` with ``states[t] != states[t-1]``: the first day of each new regime."""
    return np.flatnonzero(np.diff(states) != 0) + 1


def limit_cycle(n, period, rng, noise=0.3, share=0.8, substeps=20, burn=200):
    """Noisy limit cycle, observed in additive noise.

    Stuart-Landau oscillator (the Hopf normal form),

        dz = ((1 + i w) z - |z|^2 z) dt + noise dW,   w = 2 pi / period,

    whose attracting orbit is the unit circle. The amplitude is pinned near 1
    while the phase diffuses. It is integrated by Euler-Maruyama with ``substeps``
    steps per day and sampled daily as Re z. The result is standardized and mixed
    with i.i.d. N(0, 1) noise so the cycle carries a fraction ``share`` of the
    variance.
    """
    omega = 2.0 * np.pi / period
    dt = 1.0 / substeps
    total = n + burn
    dw = (rng.standard_normal(total * substeps) + 1j * rng.standard_normal(total * substeps))
    dw *= np.sqrt(dt / 2.0)
    z = 1.0 + 0.0j
    x = np.empty(total)
    step = 0
    for t in range(total):
        for _ in range(substeps):
            z = z + ((1.0 + 1j * omega) * z - (z.real * z.real + z.imag * z.imag) * z) * dt + noise * dw[step]
            step += 1
        x[t] = z.real
    x = standardize(x[burn:])
    return np.sqrt(share) * x + np.sqrt(1.0 - share) * rng.standard_normal(n)
