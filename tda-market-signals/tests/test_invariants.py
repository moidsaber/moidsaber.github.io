"""The mathematical facts the notebooks lean on, checked numerically."""

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from tdasig import features as F
from tdasig import simulate as S
from tdasig import surrogates as SG


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def cloud(rng, n=60, m=4):
    return rng.standard_normal((n, m))


def sort_rows(a):
    return a[np.lexsort(a.T[::-1])]


# ----------------------------------------------------------------- persistence

def test_h0_deaths_are_mst_edge_lengths(rng):
    X = cloud(rng)
    d0 = F.diagrams(X, maxdim=0)[0]
    mst = minimum_spanning_tree(squareform(pdist(X))).toarray()
    np.testing.assert_allclose(np.sort(d0[:, 1]), np.sort(mst[mst > 0]), rtol=1e-6)


def test_rips_diagrams_are_homogeneous(rng):
    X = cloud(rng)
    for a, b in zip(F.diagrams(X), F.diagrams(3.0 * X)):
        np.testing.assert_allclose(sort_rows(3.0 * a), sort_rows(b), rtol=1e-5)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_landscape_norm_matches_closed_form(rng, p):
    dgm = F.diagrams(cloud(rng))[1]
    grid = np.linspace(0.0, 1.01 * dgm[:, 1].max(), 20001)
    lam = F.landscape(dgm, grid, k=len(dgm))
    numeric = np.trapezoid((lam ** p).sum(axis=0), grid) ** (1.0 / p)
    assert numeric == pytest.approx(F.landscape_norm(dgm, p), rel=1e-4)


@pytest.mark.parametrize("p", [1, 2])
def test_landscape_norm_scales_as_c_to_one_plus_one_over_p(rng, p):
    X, c = cloud(rng), 2.5
    a = F.landscape_norm(F.diagrams(X)[1], p)
    b = F.landscape_norm(F.diagrams(c * X)[1], p)
    assert b == pytest.approx(c ** (1.0 + 1.0 / p) * a, rel=1e-5)


def test_persistence_entropy_is_scale_invariant(rng):
    X = cloud(rng)
    for k in (0, 1):
        a = F.persistence_entropy(F.diagrams(X)[k])
        b = F.persistence_entropy(F.diagrams(7.0 * X)[k])
        assert a == pytest.approx(b, rel=1e-6)


def test_betti_curve_integrates_to_total_persistence(rng):
    dgm = F.diagrams(cloud(rng))[1]
    grid = np.linspace(0.0, 1.01 * dgm[:, 1].max(), 200001)
    area = np.trapezoid(F.betti_curve(dgm, grid), grid)
    assert area == pytest.approx(F.lifetimes(dgm).sum(), rel=1e-3)


def test_persistence_image_mass_is_total_weight(rng):
    dgm = F.diagrams(cloud(rng))[1]
    p = F.lifetimes(dgm)
    big = (-10.0, 10.0)
    img = F.persistence_image(dgm, big, big, n_pix=50, sigma=0.05, weight_scale=p.max())
    assert img.sum() == pytest.approx(np.minimum(p / p.max(), 1.0).sum(), rel=1e-6)


# ----------------------------------------------------------------- windows

def test_rank_normalization_ignores_monotone_maps(rng):
    x = rng.standard_normal(100)
    np.testing.assert_array_equal(F.normalize(x, "rank"), F.normalize(np.exp(3.0 * x) + 5.0, "rank"))


def test_vol_normalization_removes_scale_and_location(rng):
    x = rng.standard_normal(100)
    np.testing.assert_allclose(F.normalize(x, "vol"), F.normalize(4.0 * x + 1.0, "vol"))


def test_delay_embedding_rows():
    E = F.delay_embed(np.arange(10.0), m=3, tau=2)
    assert E.shape == (6, 3)
    np.testing.assert_array_equal(E[0], [4.0, 2.0, 0.0])
    np.testing.assert_array_equal(E[-1], [9.0, 7.0, 5.0])


def test_make_windows():
    W, ends = F.make_windows(np.arange(25.0), 10, step=5)
    assert list(ends) == [9, 14, 19, 24]
    np.testing.assert_array_equal(W[1], np.arange(5.0, 15.0))


# ----------------------------------------------------------------- surrogates

@pytest.mark.parametrize("n", [512, 513])
def test_phase_randomization_keeps_the_periodogram(rng, n):
    x = S.garch(n, 0.1, 0.85, rng)[0]
    s = SG.phase_randomize(x, rng)
    np.testing.assert_allclose(np.abs(np.fft.rfft(s - s.mean())),
                               np.abs(np.fft.rfft(x - x.mean())), atol=1e-8)
    assert s.mean() == pytest.approx(x.mean())
    assert not np.allclose(s, x)


def test_iaaft_keeps_marginal_exactly_and_spectrum_closely(rng):
    x = S.limit_cycle(4096, period=8, rng=rng, share=0.6)
    s = SG.iaaft(x, rng)
    np.testing.assert_array_equal(np.sort(s), np.sort(x))
    a, b = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s))
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.05
    assert F.acf(s, 1) == pytest.approx(F.acf(x, 1), abs=0.02)


def test_shuffle_keeps_each_window_marginal(rng):
    W = rng.standard_normal((5, 50))
    np.testing.assert_allclose(np.sort(SG.shuffle_windows(W, rng), axis=1), np.sort(W, axis=1))


def test_garch_t_fit_recovers_parameters(rng):
    r, _ = S.garch(20_000, 0.08, 0.90, rng, nu=5)
    p = SG.fit_garch_t(r)
    assert p["alpha"] == pytest.approx(0.08, abs=0.02)
    assert p["beta"] == pytest.approx(0.90, abs=0.03)
    assert p["nu"] == pytest.approx(5.0, abs=1.5)


# ----------------------------------------------------------------- generators

def test_garch_has_unit_unconditional_variance(rng):
    r, h = S.garch(200_000, 0.08, 0.90, rng)
    assert r.var() == pytest.approx(1.0, rel=0.1)
    assert h.min() > 0


def test_regime_garch_switches_scale(rng):
    r, s = S.regime_garch(50_000, 0.08, 0.90, (1.0, 2.5), (400, 150), rng)
    sw = S.switch_dates(s)
    assert len(sw) > 50
    assert np.all(s[sw] != s[sw - 1])
    assert s.mean() == pytest.approx(150 / 550, abs=0.07)
    assert r[s == 1].std() / r[s == 0].std() == pytest.approx(2.5, rel=0.15)


def test_limit_cycle_period_and_loop(rng):
    x = S.limit_cycle(4000, period=8, rng=rng)
    freqs = np.fft.rfftfreq(len(x))[1:]
    peak = freqs[np.argmax(np.abs(np.fft.rfft(x))[1:])]
    assert 1.0 / peak == pytest.approx(8.0, rel=0.1)

    w = F.normalize(x[:100], "vol")
    loop = F.lifetimes(F.diagrams(F.delay_embed(w, 4))[1]).max()
    noise = F.lifetimes(F.diagrams(F.delay_embed(rng.standard_normal(100), 4))[1]).max()
    assert loop > 1.5 * noise


# ----------------------------------------------------------------- evaluation helpers

def test_switch_labels():
    from tdasig import evaluate as E
    ends = np.array([99, 149, 199, 249])
    assert list(E.switch_labels(ends, 100, [120])) == [0, 1, 1, 0]
    assert list(E.switch_labels(ends, 100, [60, 140])) == [1, -1, 1, 0]
    assert list(E.switch_labels(ends, 100, [95])) == [-1, 1, 0, 0]


def test_event_average():
    import pandas as pd
    from tdasig import evaluate as E
    ends = np.arange(0, 1000, 5)
    out = E.event_average(pd.DataFrame({"v": ends.astype(float)}), ends, [100, 500], [-20, 0, 10])
    assert out.loc[0, "v"] == 300.0
    assert out.loc[10, "v"] == 310.0
    assert out.loc[-20, "v"] == 280.0
