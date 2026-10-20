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


def _dataset(rng, k: int, noise: np.ndarray, separation: float, dim: int = 10, corr: float = 0.0):
    """corr: correlation between the views' noise (real representations are transforms of the same EEG, so
    their errors are correlated); each view's noise variance is unchanged."""
    y = np.repeat(np.arange(3), CLASS_SIZES)
    centres = rng.normal(0, separation, (3, dim))
    latent = centres[y] + rng.normal(0, 1, (len(y), dim))          # subject-level signal shared by all views
    common = rng.normal(0, 1, latent.shape)
    views = [latent + noise[r] * (np.sqrt(corr) * common + np.sqrt(1 - corr) * rng.normal(0, 1, latent.shape))
             for r in range(k)]
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
             spread: float = 0.0, seed: int = 0, ratio: float = RATIO_TRAIN, subject_level: bool = False,
             corr: float = 0.0) -> pd.DataFrame:
    """spread = 0: null (equal noise); spread > 0: noise levels base_noise * linspace(1 - spread, 1 + spread).
    subject_level: also test the by-subject repeated-measures F on the averaged true-class probabilities (for a
    balanced design the by-subject mixed model reduces to it)."""
    rng = np.random.default_rng(seed)
    noise = base_noise * np.linspace(1 - spread, 1 + spread, k)
    rows = []
    for i in range(n_sim):
        views, y = _dataset(rng, k, noise, separation, corr=corr)
        if subject_level:
            wide, subj = cv_scores_subject(views, y)
            s = oneway_rm(subj)
            extra = {"F_subj": s["F"], "p_subj_gg": s["p_gg"]}
        else:
            wide, extra = cv_scores(views, y), {}
        rows.append({"sim": i, "k": k, "spread": spread, "corr": corr, "mean_f1": wide.mean(),
                     "range_f1": np.ptp(wide.mean(0)), **oneway_rm(wide, ratio), **extra})
    return pd.DataFrame(rows)


def summarise(sims: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    agg = dict(n=("sim", "size"), mean_f1=("mean_f1", "mean"), mean_range=("range_f1", "mean"),
               reject_uncorrected=("p_gg", lambda p: float((p < alpha).mean())),
               reject_nb=("p_nb_gg", lambda p: float((p < alpha).mean())))
    if "p_subj_gg" in sims:
        agg["reject_subject"] = ("p_subj_gg", lambda p: float((p < alpha).mean()))
    keys = ["k", "spread"] + (["corr"] if "corr" in sims else [])
    return sims.groupby(keys).agg(**agg).reset_index()


# ---- sixth review, M1: a null simulation matched to the observed data ----

def split_stats(wide: np.ndarray) -> dict:
    """wide: splits x representations. The summaries a null simulation must reproduce."""
    c = np.corrcoef(wide, rowvar=False)
    return {"mean_f1": float(wide.mean()), "split_sd": float(wide.std(0, ddof=1).mean()),
            "rep_corr": float(c[np.triu_indices_from(c, 1)].mean()), "range_f1": float(np.ptp(wide.mean(0)))}


def observed_targets(runs: pd.DataFrame, edge: str | None = None) -> dict:
    """runs: load_runs rows of the main grid. Per split, each representation's macro-F1 averaged over graphs, or
    for one graph type only (one model per representation and split, as in the simulation)."""
    if edge is not None:
        runs = runs[runs.edge == edge]
    wide = runs.groupby(["unit", "pipeline"])["macro_f1"].mean().unstack().to_numpy()
    return split_stats(wide)


def calibrate(targets: dict, separations=(0.35, 0.5, 0.65, 0.8, 1.0), noises=(0.5, 1.0, 1.5, 2.0),
              corrs=(0.0, 0.4, 0.7, 0.9), n_sim: int = 6, k: int = 7, seed: int = 0) -> pd.DataFrame:
    """Grid over the null simulation's free parameters; distance to the observed mean macro-F1, split-to-split
    SD and between-representation correlation of split scores (scaled by rough tolerances)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sep in separations:
        for nz in noises:
            for c in corrs:
                st = [split_stats(cv_scores(*_dataset(rng, k, np.full(k, nz), sep, corr=c))) for _ in range(n_sim)]
                m = {key: float(np.mean([s[key] for s in st])) for key in ("mean_f1", "split_sd", "rep_corr")}
                d = (((m["mean_f1"] - targets["mean_f1"]) / 0.02) ** 2 + ((m["split_sd"] - targets["split_sd"]) / 0.01) ** 2
                     + ((m["rep_corr"] - targets["rep_corr"]) / 0.05) ** 2)
                rows.append({"separation": sep, "noise": nz, "corr": c, **m, "distance": float(d)})
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)


def calibrated_null(targets: dict, separation: float, noise: float, corr: float, n_sim: int = 300, k: int = 7,
                    seed: int = 1, ratio: float = RATIO_TRAIN) -> tuple[pd.DataFrame, dict]:
    """Null datasets with the calibrated parameters; where does the observed spread fall?"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sim):
        wide = cv_scores(*_dataset(rng, k, np.full(k, noise), separation, corr=corr))
        rows.append({"sim": i, **split_stats(wide), **oneway_rm(wide, ratio)})
    sims = pd.DataFrame(rows)
    obs = targets["range_f1"]
    # the simulation cannot reach the observed split SD at this mean F1, so also compare the spread in SD units
    z, obs_z = sims.range_f1 / sims.split_sd, obs / targets["split_sd"]
    summary = {"n_sim": n_sim, "observed_range": obs, "null_range_mean": float(sims.range_f1.mean()),
               "null_range_q95": float(sims.range_f1.quantile(0.95)),
               "p_range": float((1 + (sims.range_f1 >= obs).sum()) / (1 + n_sim)),
               "observed_std_range": float(obs_z), "null_std_range_mean": float(z.mean()),
               "null_std_range_q95": float(z.quantile(0.95)),
               "p_std_range": float((1 + (z >= obs_z).sum()) / (1 + n_sim)),
               "null_mean_f1": float(sims.mean_f1.mean()), "null_split_sd": float(sims.split_sd.mean()),
               "null_rep_corr": float(sims.rep_corr.mean()),
               "reject_uncorrected": float((sims.p_gg < 0.05).mean()), "reject_nb": float((sims.p_nb_gg < 0.05).mean())}
    return sims, summary
