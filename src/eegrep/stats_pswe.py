"""RQ6: does the PSWE burden (BBB-dysfunction-associated marker) differ between AD,
FTD and CN, and is it more than generic EEG slowing?

Analysis only: MMSE appears here for the within-patient correlation and is never a
model input. All p-value families are Holm-corrected.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .cache import FeatureCache


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def dunn(values: np.ndarray, groups: np.ndarray, names: dict) -> list[dict]:
    """Dunn's post-hoc test on mid-ranks with tie correction, Holm-adjusted."""
    r = stats.rankdata(values)
    n = len(values)
    ties = np.unique(values, return_counts=True)[1]
    tie_term = (ties ** 3 - ties).sum() / (12 * (n - 1))
    out = []
    for a, b in itertools.combinations(sorted(names), 2):
        ra, rb = r[groups == a], r[groups == b]
        se = np.sqrt((n * (n + 1) / 12 - tie_term) * (1 / len(ra) + 1 / len(rb)))
        z = (ra.mean() - rb.mean()) / se
        u = stats.mannwhitneyu(values[groups == a], values[groups == b])
        rbc = 1 - 2 * u.statistic / (len(ra) * len(rb))          # rank-biserial r
        out.append({"pair": f"{names[a]} vs {names[b]}", "z": float(z),
                    "p": float(2 * stats.norm.sf(abs(z))), "rank_biserial": float(rbc)})
    for row, padj in zip(out, holm([o["p"] for o in out])):
        row["p_holm"] = padj
    return out


def subject_table(cache_dir: str | Path, participants: pd.DataFrame) -> pd.DataFrame:
    cache = FeatureCache(cache_dir)
    ps = pd.read_csv(Path(cache_dir) / "pswe_subject.csv")
    ev = pd.read_csv(Path(cache_dir) / "pswe_events.csv")
    per_subj = ps.groupby("subject").agg(rate_per_min=("rate_per_min", "mean"),
                                         time_fraction=("time_fraction", "mean"),
                                         duration_s=("duration_s", "first"))
    per_subj["n_events"] = ev.groupby("subject").size().reindex(per_subj.index, fill_value=0)
    p1 = cache.X("P1")
    slow = pd.Series((p1[..., 0] + p1[..., 1]).mean(-1)).groupby(cache.subject).mean()   # rel δ+θ power
    per_subj["rel_delta_theta"] = slow.reindex(per_subj.index)
    return per_subj.join(participants.set_index("subject")[["group", "label", "age", "sex_male", "mmse"]])


def rq6(df: pd.DataFrame, class_names: list[str]) -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    names = dict(enumerate(class_names))
    g = df["label"].to_numpy()
    out = {"n": int(len(df)), "descriptives": {}}
    for metric in ("rate_per_min", "time_fraction"):
        v = df[metric].to_numpy()
        kw = stats.kruskal(*[v[g == c] for c in names])
        k, n = len(names), len(v)
        out[metric] = {"kruskal_H": float(kw.statistic), "p": float(kw.pvalue),
                       "epsilon_sq": float((kw.statistic - k + 1) / (n - k)),
                       "dunn": dunn(v, g, names)}
        out["descriptives"][metric] = {names[c]: {"median": float(np.median(v[g == c])),
                                                  "iqr": [float(np.percentile(v[g == c], 25)),
                                                          float(np.percentile(v[g == c], 75))]}
                                       for c in names}

    d = df.assign(group=pd.Categorical(df["label"].map(names), categories=class_names),
                  log_dur_min=np.log(df["duration_s"] / 60.0))
    # NB2 regression of event counts (dispersion estimated), exposure = recording minutes,
    # adjusted for age and sex; the second model also adjusts for generic slowing (δ+θ power).
    for name, formula in {"nb_glm_adjusted": "n_events ~ C(group, Treatment('CN')) + age + sex_male",
                          "nb_glm_adjusted_slowing": "n_events ~ C(group, Treatment('CN')) + age + sex_male"
                                                     " + rel_delta_theta"}.items():
        fit = smf.negativebinomial(formula, data=d, exposure=np.exp(d["log_dur_min"])).fit(disp=0, maxiter=500)
        out[name] = {"rate_ratio": {k: float(np.exp(v)) for k, v in fit.params.items()},
                     "p": {k: float(v) for k, v in fit.pvalues.items()},
                     "ci95": {k: [float(np.exp(a)), float(np.exp(b))] for k, (a, b) in fit.conf_int().iterrows()}}

    # Is PSWE just slowing?  Spearman with relative δ+θ power, and partial (age, sex) correlation.
    rho = stats.spearmanr(df["rate_per_min"], df["rel_delta_theta"])
    out["pswe_vs_slowing_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
    rk = df[["rate_per_min", "rel_delta_theta", "age", "sex_male"]].rank()
    z = sm.add_constant(rk[["age", "sex_male"]])
    res = [sm.OLS(rk[c], z).fit().resid for c in ("rate_per_min", "rel_delta_theta")]
    pr = stats.pearsonr(*res)
    out["pswe_vs_slowing_partial_rank"] = {"r": float(pr.statistic), "p": float(pr.pvalue)}

    # Within patients (AD + FTD): PSWE rate vs MMSE (CN have MMSE 30, so excluded).
    pat = df[df["label"] != class_names.index("CN")]
    s = stats.spearmanr(pat["rate_per_min"], pat["mmse"])
    out["pswe_vs_mmse_patients"] = {"rho": float(s.statistic), "p": float(s.pvalue), "n": int(len(pat))}
    return out


def channel_topography(cache_dir: str | Path, participants: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Median per-channel PSWE rate per group (for topomaps, Figure 5)."""
    ps = pd.read_csv(Path(cache_dir) / "pswe_subject.csv").merge(
        participants[["subject", "label"]], on="subject")
    ps["group"] = ps["label"].map(dict(enumerate(class_names)))
    return ps.pivot_table(index="channel", columns="group", values="rate_per_min", aggfunc="median")


def write_rq6(cache_dir, participants, class_names, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = subject_table(cache_dir, participants)
    df.to_csv(out / "pswe_subject_table.csv")
    channel_topography(cache_dir, participants, class_names).to_csv(out / "pswe_topography.csv")
    res = rq6(df, class_names)
    (out / "rq6_pswe_stats.json").write_text(json.dumps(res, indent=1))
    return res
