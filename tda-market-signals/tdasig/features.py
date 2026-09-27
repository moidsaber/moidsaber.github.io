"""Window -> normalize -> delay-embed -> Rips persistence -> features.

Conventions
-----------
* A *window* is ``w`` consecutive returns. Its point cloud is the delay embedding
  ``v_t = (x_t, x_{t-tau}, ..., x_{t-(m-1)tau})``, which gives ``n = w - (m-1) tau`` points in R^m.
* Persistence is Vietoris-Rips (ripser) in dimensions 0 and 1. In Rips every H0
  class is born at 0, and the finite H0 deaths are exactly the edge lengths of the
  Euclidean minimum spanning tree. The one infinite H0 bar is dropped.
* Diagrams are ``(k, 2)`` arrays of (birth, death); lifetimes ``l = death - birth``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ripser import ripser
from scipy.special import ndtr
from scipy.stats import kurtosis, rankdata

NORMALIZATIONS = ("raw", "vol", "rank")

TDA_SCALARS = ("H0_total", "H0_max", "H0_cv", "H0_entropy",
               "H1_total", "H1_max", "H1_entropy", "H1_L1")
CLASSICAL = ("rv", "kurt", "geary", "maxabs", "acf1", "acf_abs", "fisher_g", "vol_ratio")


@dataclass(frozen=True)
class Config:
    """Pipeline parameters: window length, embedding dimension, delay, max homology degree."""
    w: int = 100
    m: int = 4
    tau: int = 1
    maxdim: int = 1

    @property
    def n_points(self):
        return self.w - (self.m - 1) * self.tau


# --------------------------------------------------------------------------- windows

def normalize(x, how):
    """Per-window normalization.

    raw  : unchanged.
    vol  : divide by the window's realized vol (and demean; Rips is translation
           invariant, so only the scale matters). Removes scale, keeps shape.
    rank : ranks / (w + 1), the empirical copula. Removes the marginal entirely,
           so only dependence is left. An i.i.d. window becomes a uniformly random
           permutation, whatever its marginal law.
    """
    x = np.asarray(x, dtype=float)
    if how == "raw":
        return x
    if how == "vol":
        return (x - x.mean()) / x.std()
    if how == "rank":
        return rankdata(x) / (len(x) + 1.0)
    raise ValueError(f"unknown normalization {how!r}")


def delay_embed(x, m, tau=1):
    """Rows ``v_t = (x_t, x_{t-tau}, ..., x_{t-(m-1)tau})`` for every t with a full history."""
    x = np.asarray(x, dtype=float)
    start = (m - 1) * tau
    n = len(x) - start
    if n <= 0:
        raise ValueError("series shorter than the embedding span")
    return np.column_stack([x[start - k * tau: start - k * tau + n] for k in range(m)])


def make_windows(x, w, step=None):
    """Windows of length ``w`` ending at ``w-1, w-1+step, ...`` (non-overlapping by default).

    Returns ``(W, ends)``: ``W[j] = x[ends[j]-w+1 : ends[j]+1]``.
    """
    step = w if step is None else step
    ends = np.arange(w - 1, len(x), step)
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=float), w)
    return view[ends - w + 1], ends


# --------------------------------------------------------------------------- persistence

def diagrams(cloud, maxdim=1):
    """Rips persistence diagrams ``[H0, H1, ...]`` with the infinite H0 bar removed."""
    dgms = ripser(cloud, maxdim=maxdim)["dgms"]
    dgms[0] = dgms[0][np.isfinite(dgms[0][:, 1])]
    return dgms


def lifetimes(dgm):
    return dgm[:, 1] - dgm[:, 0] if len(dgm) else np.zeros(0)


def persistence_entropy(dgm):
    """E = -sum p_i log p_i with p_i = l_i / sum l. Invariant under rescaling the diagram."""
    ell = lifetimes(dgm)
    ell = ell[ell > 0]
    if len(ell) == 0:
        return 0.0
    p = ell / ell.sum()
    return float(-(p * np.log(p)).sum())


def landscape_norm(dgm, p=1):
    """Exact L^p norm of the full persistence landscape, summed over all levels k.

    At each t the values {lambda_k(t)}_k are the tent values {Lambda_i(t)}_i sorted,
    so sum_k lambda_k(t)^p = sum_i Lambda_i(t)^p. Integrating one tent gives

        ||lambda||_p^p = sum_i l_i^(p+1) / (2^p (p+1)).

    The norm therefore depends on the lifetimes only, and scales as c^(1+1/p).
    """
    ell = lifetimes(dgm)
    return float((np.sum(ell ** (p + 1)) / (2.0 ** p * (p + 1))) ** (1.0 / p))


def landscape(dgm, grid, k=3):
    """First ``k`` landscape functions lambda_1..lambda_k on ``grid``, shape (k, len(grid))."""
    out = np.zeros((k, len(grid)))
    if len(dgm) == 0:
        return out
    tents = np.maximum(0.0, np.minimum(grid[None, :] - dgm[:, :1], dgm[:, 1:2] - grid[None, :]))
    top = -np.sort(-tents, axis=0)[:k]
    out[: len(top)] = top
    return out


def betti_curve(dgm, grid):
    """beta(t) = #{i : b_i <= t < d_i}, i.e. how many features are alive at scale t."""
    if len(dgm) == 0:
        return np.zeros(len(grid))
    alive = (dgm[:, :1] <= grid[None, :]) & (grid[None, :] < dgm[:, 1:2])
    return alive.sum(axis=0).astype(float)


def persistence_image(dgm, birth_range, pers_range, n_pix, sigma, weight_scale):
    """Persistence image (Adams et al. 2017) on the (birth, persistence) plane.

    rho(x, y) = sum_i w(p_i) phi_i(x, y), where phi_i is an isotropic Gaussian with
    std ``sigma`` centred at (b_i, p_i). The weight w(p) = min(p / weight_scale, 1)
    vanishes on the diagonal and is Lipschitz. Each pixel holds the exact integral
    of rho over the pixel. Rows index persistence and columns index birth.
    """
    img = np.zeros((n_pix, n_pix))
    if len(dgm) == 0:
        return img
    b, p = dgm[:, 0], dgm[:, 1] - dgm[:, 0]
    wts = np.minimum(p / weight_scale, 1.0)
    bx = np.linspace(*birth_range, n_pix + 1)
    py = np.linspace(*pers_range, n_pix + 1)
    gx = np.diff(ndtr((bx[None, :] - b[:, None]) / sigma), axis=1)
    gy = np.diff(ndtr((py[None, :] - p[:, None]) / sigma), axis=1)
    return np.einsum("i,iy,ix->yx", wts, gy, gx)


def tda_summaries(dgms):
    """Scalar summaries of the H0 and H1 diagrams (see ``TDA_SCALARS``)."""
    l0, l1 = lifetimes(dgms[0]), lifetimes(dgms[1])
    return {
        "H0_total": l0.sum(),                    # length of the MST (Steele's functional)
        "H0_max": l0.max(),                      # longest MST edge: the most isolated point
        "H0_cv": l0.std() / l0.mean(),           # spread of MST edges: two-scale structure
        "H0_entropy": persistence_entropy(dgms[0]),
        "H1_total": l1.sum(),
        "H1_max": l1.max() if len(l1) else 0.0,  # Perea-Harer-style periodicity score
        "H1_entropy": persistence_entropy(dgms[1]),
        "H1_L1": landscape_norm(dgms[1], 1),     # the Gidea-Katz statistic, = sum l^2 / 4
    }


# --------------------------------------------------------------------------- classical baselines

def acf(y, lag):
    d = y - y.mean()
    return float(d[:-lag] @ d[lag:] / (d @ d))


def classical_features(y):
    """Cheap non-topological statistics of the (normalized) window ``y``.

    rv        realized vol (only varies for raw windows)
    kurt      excess kurtosis (tails / scale mixing)
    geary     mean |y - mean| / sd, Geary's ratio: a low-variance robust kurtosis
              (sqrt(2/pi) ~ 0.80 for a Gaussian, lower for fat tails)
    maxabs    largest standardized deviation (the single most extreme day)
    acf1      lag-1 autocorrelation (linear dependence)
    acf_abs   mean lag-1..5 autocorrelation of |y - mean| (volatility clustering)
    fisher_g  largest periodogram ordinate / total (Fisher's test for a hidden periodicity)
    vol_ratio |log(sd of 2nd half / sd of 1st half)| (a variance change inside the window;
              the only statistic here that uses the order of the whole window)
    """
    d = y - y.mean()
    a = np.abs(d)
    half = len(y) // 2
    pgram = np.abs(np.fft.rfft(d))[1:(len(y) + 1) // 2] ** 2
    return {
        "rv": y.std(),
        "kurt": kurtosis(y),
        "geary": a.mean() / y.std(),
        "maxabs": a.max() / y.std(),
        "acf1": acf(y, 1),
        "acf_abs": float(np.mean([acf(a, k) for k in range(1, 6)])),
        "fisher_g": pgram.max() / pgram.sum(),
        "vol_ratio": abs(np.log(y[half:].std() / y[:half].std())),
    }


# --------------------------------------------------------------------------- vectorization on fixed grids

@dataclass(frozen=True)
class Grids:
    """Fixed evaluation grids, shared by every window so the vectors are comparable."""
    t: np.ndarray            # filtration values for landscapes and Betti curves
    birth_range: tuple       # persistence image, H1 births
    pers_range: tuple        # persistence image, H1 persistence
    n_pix: int
    sigma: float
    weight_scale: float


def make_grids(dgm_list, n_grid=50, n_pix=12, q=0.995):
    """Build grids covering the ``q``-quantiles of the pooled diagrams in ``dgm_list``."""
    h0_death = np.array([d[0][:, 1].max() for d in dgm_list])
    h1 = np.vstack([d[1] for d in dgm_list if len(d[1])])
    t_max = max(np.quantile(h0_death, q), np.quantile(h1[:, 1], q))
    b_max = np.quantile(h1[:, 0], q)
    p_max = np.quantile(h1[:, 1] - h1[:, 0], q)
    return Grids(t=np.linspace(0.0, t_max, n_grid), birth_range=(0.0, b_max),
                 pers_range=(0.0, p_max), n_pix=n_pix, sigma=p_max / n_pix,
                 weight_scale=p_max)


def vectorize(dgms, grids, k=3):
    """Full-vector representations of one window's diagrams, as a dict of 1-D arrays.

    landscape  lambda_1..lambda_k for H0 and H1 on ``grids.t``
    betti      beta_0 and beta_1 on ``grids.t``
    image      H1 persistence image. In Rips every H0 birth is 0, so an H0 image
               would collapse to a 1-D density of death times, which the other
               two representations already carry.
    entropy    persistence entropy of H0 and H1
    """
    return {
        "landscape": np.concatenate([landscape(dgms[0], grids.t, k).ravel(),
                                     landscape(dgms[1], grids.t, k).ravel()]),
        "betti": np.concatenate([betti_curve(dgms[0], grids.t), betti_curve(dgms[1], grids.t)]),
        "image": persistence_image(dgms[1], grids.birth_range, grids.pers_range,
                                   grids.n_pix, grids.sigma, grids.weight_scale).ravel(),
        "entropy": np.array([persistence_entropy(dgms[0]), persistence_entropy(dgms[1])]),
    }


def vectorize_all(dgm_list, grids, k=3):
    """Stack ``vectorize`` over windows: dict of (n_windows, dim) arrays."""
    vecs = [vectorize(d, grids, k) for d in dgm_list]
    return {name: np.vstack([v[name] for v in vecs]) for name in vecs[0]}


# --------------------------------------------------------------------------- the pipeline

def window_features(window, norm, cfg, keep_diagrams=False):
    """All scalar features of one raw window under normalization ``norm``."""
    y = normalize(window, norm)
    dgms = diagrams(delay_embed(y, cfg.m, cfg.tau), cfg.maxdim)
    feats = {**tda_summaries(dgms), **classical_features(y)}
    return feats, (dgms if keep_diagrams else None)


def compute_features(W, norm, cfg, keep_diagrams=False, n_jobs=-1):
    """Scalar features for every row of ``W`` (in parallel).

    Returns ``(df, dgms)``: a DataFrame with one row per window, and the list of
    diagrams when ``keep_diagrams`` is set (otherwise None).
    """
    res = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(window_features)(w, norm, cfg, keep_diagrams) for w in W)
    df = pd.DataFrame([r[0] for r in res])
    return df, ([r[1] for r in res] if keep_diagrams else None)
