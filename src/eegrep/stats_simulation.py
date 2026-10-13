"""Calibration of the Nadeau-Bengio-adjusted omnibus F (third internal review, S1).

For k = 2 the adjusted F equals the squared Nadeau-Bengio t. For k > 2 it is our generalisation, so we check its
false-positive rate and power by simulation under the study's own resampling scheme: 88 subjects in three classes
(36/23/29), 5 x 5 repeated stratified subject-level cross-validation with 20% of each training set held out (as
the validation subjects are), and k "representations" that are noisy views of one latent class signal. Under the
null every representation has the same noise level, so their expected scores are equal and any significant
omnibus test is a false positive; under the alternative the noise level differs between representations.
The split-level repeated-measures F (uncorrected, Greenhouse-Geisser) and the adjusted F are applied to the
per-split macro-F1, exactly as for the main grid (a main effect in the two-way design is the one-way test on the
marginal means, so the one-way version is what is checked here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .stats_grid import RATIO_TRAIN, _gg_epsilon, _orthonormal_contrasts

CLASS_SIZES = (36, 23, 29)


def oneway_rm(wide: np.ndarray, ratio: float = RATIO_TRAIN) -> dict:
    """wide: n_units x k scores. Uncorrected (GG) and Nadeau-Bengio-adjusted repeated-measures F."""
    n, k = wide.shape
    m = wide.mean()
    ss_eff = n * ((wide.mean(0) - m) ** 2).sum()
    resid = wide - wide.mean(1, keepdims=True) - wide.mean(0, keepdims=True) + m
    ss_err = (resid ** 2).sum()
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = (ss_eff / df1) / (ss_err / df2)
    eps = min(1.0, _gg_epsilon(wide, _orthonormal_contrasts(k)))
    f_nb = f / (1.0 + n * ratio)
    return {"F": f, "p_gg": float(stats.f.sf(f, df1 * eps, df2 * eps)),
            "F_nb": f_nb, "p_nb_gg": float(stats.f.sf(f_nb, df1 * eps, df2 * eps))}


def _dataset(rng, k: int, noise: np.ndarray, separation: float, dim: int = 10):
    y = np.repeat(np.arange(3), CLASS_SIZES)
    centres = rng.normal(0, separation, (3, dim))
    latent = centres[y] + rng.normal(0, 1, (len(y), dim))          # subject-level signal shared by all views
    views = [latent + rng.normal(0, noise[r], latent.shape) for r in range(k)]
    return views, y


def cv_scores(views, y, seeds=range(5), n_folds: int = 5, val_frac: float = 0.2) -> np.ndarray:
    """25 splits x k macro-F1, every representation evaluated on the same splits (paired, as in the study)."""
    out = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for f, (trv, te) in enumerate(skf.split(views[0], y)):
            tr, _ = train_test_split(trv, test_size=val_frac, stratify=y[trv], random_state=seed * 1000 + f)
            row = []
            for v in views:
                clf = LogisticRegression(max_iter=500, C=0.5).fit(v[tr], y[tr])
                row.append(f1_score(y[te], clf.predict(v[te]), average="macro"))
            out.append(row)
    return np.asarray(out)


def cv_scores_subject(views, y, seeds=range(5), n_folds: int = 5, val_frac: float = 0.2):
    """As cv_scores, plus each subject's probability of its true class averaged over the repeats (subjects x k),
    the quantity the by-subject mixed model analyses."""
    n, k = len(y), len(views)
    split_scores, p_true, counts = [], np.zeros((n, k)), np.zeros(n)
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for f, (trv, te) in enumerate(skf.split(views[0], y)):
            tr, _ = train_test_split(trv, test_size=val_frac, stratify=y[trv], random_state=seed * 1000 + f)
            row = []
            for r, v in enumerate(views):
                clf = LogisticRegression(max_iter=500, C=0.5).fit(v[tr], y[tr])
                prob = clf.predict_proba(v[te])
                row.append(f1_score(y[te], prob.argmax(1), average="macro"))
                p_true[te, r] += prob[np.arange(len(te)), y[te]]
            counts[te] += 1
            split_scores.append(row)
    return np.asarray(split_scores), p_true / counts[:, None]


def simulate(n_sim: int = 300, k: int = 7, separation: float = 0.35, base_noise: float = 1.5,
             spread: float = 0.0, seed: int = 0, ratio: float = RATIO_TRAIN, subject_level: bool = False) -> pd.DataFrame:
    """spread = 0: null (equal noise); spread > 0: noise levels base_noise * linspace(1 - spread, 1 + spread).
    subject_level: also test the by-subject repeated-measures F on the averaged true-class probabilities (for a
    balanced design the by-subject mixed model reduces to it)."""
    rng = np.random.default_rng(seed)
    noise = base_noise * np.linspace(1 - spread, 1 + spread, k)
    rows = []
    for i in range(n_sim):
        views, y = _dataset(rng, k, noise, separation)
        if subject_level:
            wide, subj = cv_scores_subject(views, y)
            s = oneway_rm(subj)
            extra = {"F_subj": s["F"], "p_subj_gg": s["p_gg"]}
        else:
            wide, extra = cv_scores(views, y), {}
        rows.append({"sim": i, "k": k, "spread": spread, "mean_f1": wide.mean(),
                     "range_f1": np.ptp(wide.mean(0)), **oneway_rm(wide, ratio), **extra})
    return pd.DataFrame(rows)


def summarise(sims: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    agg = dict(n=("sim", "size"), mean_f1=("mean_f1", "mean"), mean_range=("range_f1", "mean"),
               reject_uncorrected=("p_gg", lambda p: float((p < alpha).mean())),
               reject_nb=("p_nb_gg", lambda p: float((p < alpha).mean())))
    if "p_subj_gg" in sims:
        agg["reject_subject"] = ("p_subj_gg", lambda p: float((p < alpha).mean()))
    return sims.groupby(["k", "spread"]).agg(**agg).reset_index()
