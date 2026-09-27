"""Evaluation: discrimination, surrogate tests, nested forecast comparison, run logging."""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _is_constant(v):
    v = np.asarray(v, dtype=float)
    return np.ptp(v) <= 1e-9 * max(1.0, np.abs(v).max())


def auc(pos, neg):
    """P(pos > neg) + P(pos = neg) / 2 for independent draws: the Mann-Whitney AUC."""
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def response_matrix(samples, null, features):
    """Signed effect 2*AUC - 1 of each feature, for each sample against the null sample.

    ``samples`` maps a name to a DataFrame of window features. Returns a features x
    samples DataFrame. It reads +1 when every window in the sample exceeds every null
    window, 0 when the two are indistinguishable, and -1 in the opposite case. Features
    that are constant by construction come back as NaN.
    """
    out = pd.DataFrame(index=list(features), columns=list(samples), dtype=float)
    for name, df in samples.items():
        for f in features:
            if _is_constant(np.r_[df[f], null[f]]):
                continue
            out.loc[f, name] = 2.0 * auc(df[f].to_numpy(), null[f].to_numpy()) - 1.0
    return out


def cv_auc(X, y, groups=None, model="boosting", n_splits=5, seed=0):
    """Out-of-fold ROC AUC of a classifier trained on the feature block ``X``.

    model="boosting"  gradient-boosted trees. This asks whether *any* function of X
                      separates the classes, which is the right question when
                      comparing feature sets.
    model="logistic"  L2 logistic regression on standardized X. It is linear, so a
                      block can win just by being a better basis (e.g. a Betti curve
                      is a nonlinear basis in scale) rather than by carrying more
                      information.

    With ``groups`` the folds are grouped. Use contiguous time blocks for
    overlapping windows, so neighbours of a test window never sit in the training
    fold.
    """
    if model == "boosting":
        est = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=8,
                                             min_samples_leaf=20, l2_regularization=1.0,
                                             random_state=seed)
    elif model == "logistic":
        est = make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 5), cv=3, scoring="roc_auc", max_iter=5000))
    else:
        raise ValueError(model)
    if groups is None:
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    else:
        cv = GroupKFold(n_splits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # sklearn API-transition notices
        p = cross_val_predict(est, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


# --------------------------------------------------------------------------- change points

def switch_labels(ends, w, switches, min_frac=0.2):
    """Label windows ``[end-w+1, end]`` by the regime switches they contain.

    Returns 1 if a window contains exactly one switch with at least ``min_frac`` of
    its days on each side, 0 if it contains none, and -1 (ambiguous) otherwise.
    """
    ends = np.asarray(ends)
    switches = np.asarray(switches)
    lab = np.full(len(ends), -1)
    for j, e in enumerate(ends):
        inside = switches[(switches > e - w + 1) & (switches <= e)]
        if len(inside) == 0:
            lab[j] = 0
        elif len(inside) == 1 and min_frac <= (inside[0] - (e - w + 1)) / w <= 1 - min_frac:
            lab[j] = 1
    return lab


def event_average(values, ends, events, lags):
    """Average ``values`` (one per window end) over events, as a function of ``end - event``.

    ``values`` is a DataFrame (or 2-D array) aligned with ``ends``. Returns a
    DataFrame indexed by lag. For each event and lag it uses the window whose end is
    nearest to ``event + lag``, provided it is within half a step.
    """
    values = pd.DataFrame(values).reset_index(drop=True)
    ends = np.asarray(ends)
    tol = np.median(np.diff(ends)) / 2.0
    rows = []
    for ev in events:
        for lag in lags:
            j = int(np.argmin(np.abs(ends - (ev + lag))))
            if abs(ends[j] - (ev + lag)) <= tol:
                rows.append(values.iloc[j].rename(lag))
    return pd.DataFrame(rows).groupby(level=0).mean()


def surrogate_test(obs, sur):
    """z-score and two-sided rank p-value of statistic ``obs`` against surrogate draws ``sur``."""
    sur = np.asarray(sur, dtype=float)
    if _is_constant(np.r_[sur, obs]):
        return np.nan, np.nan
    z = (obs - sur.mean()) / sur.std(ddof=1)
    k = min((sur >= obs).sum(), (sur <= obs).sum())
    return z, min(1.0, 2.0 * (k + 1) / (len(sur) + 1))


# --------------------------------------------------------------------------- forecasting

def qlike(realized, forecast):
    """QLIKE loss for variance forecasts, which is robust to noise in the realized proxy."""
    ratio = np.asarray(realized) / np.asarray(forecast)
    return ratio - np.log(ratio) - 1.0


def newey_west_var(x, lag):
    """Long-run variance of ``x`` with a Bartlett kernel (Newey-West)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    for k in range(1, lag + 1):
        v += 2.0 * (1.0 - k / (lag + 1.0)) * (x[k:] @ x[:-k]) / n
    return v


def clark_west(y, f_small, f_big, lag=0):
    """Clark-West (2007) test that a larger nested model improves squared-error forecasts.

    Returns ``(t, p)`` with a one-sided p-value for H1: the big model is better.
    The adjustment term removes the noise that estimating extra parameters adds
    under the null, which a plain Diebold-Mariano test mishandles for nested models.
    """
    f = (y - f_small) ** 2 - ((y - f_big) ** 2 - (f_small - f_big) ** 2)
    t = f.mean() / np.sqrt(newey_west_var(f, lag) / len(f))
    return float(t), float(1.0 - norm.cdf(t))


# --------------------------------------------------------------------------- run log

def log_run(path, name, config, metrics):
    """Append one JSON line recording a configuration and its headline results.

    Every configuration you try goes in the log, including the ones that go nowhere.
    That is what makes the final multiple-testing count honest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name,
           "config": config, "metrics": metrics}
    with path.open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
