# TDA market signals

Do topological features of recent returns carry information about future risk that volatility models don't already have?

This project answers that in two steps:

1. Establish, on synthetic data with known ground truth, what each topological feature actually responds to.
2. Take that to real markets with the right normalization, the right nulls and the right baselines.

## Notebooks

| notebook | question | status |
|---|---|---|
| [`01_synthetic_mechanisms.ipynb`](notebooks/01_synthetic_mechanisms.ipynb) | Which features respond to fat tails, vol clustering, regime switches and cycles? How much is just scale? Is anything left beyond GARCH, and beyond cheap classical statistics? | done |
| `02_real_data` | Gidea–Katz replication, raw vs vol-normalized, then walk-forward vol and drawdown forecasts against HAR and GARCH | next |

### Headline results from notebook 01

- **Raw features are mostly scale.** For a feature of homogeneity degree $k$, $\log F_{\text{raw}} = k\log\hat\sigma_w + \log F_{\text{vol}}$ holds exactly. On GARCH windows, 66% of the variance of the log Gidea–Katz statistic is window vol.
- **The full-landscape $L^p$ norm is a power sum of bar lengths:** $\lVert\lambda\rVert_p^p = \sum_i \ell_i^{p+1} / (2^p (p+1))$. It forgets births, and the $c^{1+1/p}$ scaling follows at once.
- **What each mechanism moves.** Fat tails, clustering and regime mixing show up in $H_0$ as two-scale structure. Only a genuine cycle moves $H_1$ up.
- **The classical statistics do as well or better.** For each mechanism, the classical statistic built for it matches or beats every TDA block (summaries, landscapes, Betti curves, persistence images). TDA adds no forecasting skill over a GARCH baseline on GARCH data.
- **Regime switching looks like GARCH at realistic sample sizes.** At 5,000 days no feature separates it from a fitted GARCH-t. At 40,000 days only the normalized $H_1$ summaries start to (z ≈ 3). That is the one place topology leads, and it argues for pooling data across assets.

## Layout

```
tdasig/
  simulate.py     ground-truth generators: Gaussian, Student-t, GARCH, regime GARCH, Stuart-Landau limit cycle
  features.py     normalize -> delay-embed -> Rips persistence -> summaries, landscapes, Betti curves, images
  surrogates.py   shuffle, Fourier and IAAFT surrogates, GARCH(1,1)-t fit and simulation
  baselines.py    realized-variance targets, HAR design, GARCH filter and forecasts
  evaluate.py     AUC and cross-validated AUC, surrogate z-tests, QLIKE, Clark-West, change-point labels, run log
  plotting.py     figure style
tests/            the mathematical invariants the notebooks rely on (pytest)
notebooks/        the studies, committed with outputs
results/runs.jsonl  one line per notebook run: configuration plus headline metrics
```

## Running it

```bash
pip install -r requirements.txt
pytest                                   # about 3 seconds
jupyter lab notebooks/                   # notebook 01 takes about 6 minutes on 4 cores
```

Every experiment draws from its own named random stream derived from one master seed, so reruns are exact. Each run of a notebook appends its configuration and headline numbers to `results/runs.jsonl`. Keep every configuration you try in that log, including the dead ends, so the final multiple-testing count is honest.

Tested with Python 3.11, numpy 2.4, scipy 1.17, pandas 3.0, scikit-learn 1.9, ripser 0.6.15, arch 8.0 and matplotlib 3.11.
