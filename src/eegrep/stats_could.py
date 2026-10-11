"""Could-fix analyses after the second internal review (pre-specified in IMPLEMENTATION_PLAN.md, NB18).

1. Severity-stratified results. Analysis only: MMSE is read here for stratification and never reaches a
   model. Patients are split at the conventional mild cut-point (MMSE >= 21 mild, <= 20 moderate-to-severe)
   and each patient's outcomes are averaged over the repeats of a cell.
2. EEGNet with the cross-subject dropout of 0.25 (the NB06 baseline used the shared 0.5), compared with the
   original EEGNet and the GCN in its own Holm family, so the pre-specified deep-baseline family is unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stats_grid import RATIO_TRAIN, nadeau_bengio
from .stats_pswe import holm

MILD_MIN_MMSE = 21


def subject_predictions(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, held-out subject), with the predicted class."""
    rows = []
    for r in runs.itertuples(index=False):
        prob = np.asarray(r.preds["prob"])
        for s, y, p in zip(r.preds["subjects"], r.preds["y"], prob):
            rows.append((s, int(y), float(p[int(y)]), int(p.argmax()), r.pipeline, r.edge, f"{r.seed}-{r.fold}"))
    return pd.DataFrame(rows, columns=["subject", "true_class", "p_true", "pred", "pipeline", "edge", "unit"])


def _per_subject(long: pd.DataFrame, cn_label: int) -> pd.DataFrame:
    return long.assign(correct=(long.pred == long.true_class).astype(float),
                       pred_cn=(long.pred == cn_label).astype(float)).groupby(
        ["subject", "true_class"], as_index=False)[["p_true", "correct", "pred_cn"]].mean()


def stratify(per_subj: pd.DataFrame, participants: pd.DataFrame, mild_min: int = MILD_MIN_MMSE) -> pd.DataFrame:
    df = per_subj.merge(participants[["subject", "mmse"]], on="subject", how="left")
    df["stratum"] = np.where(df["mmse"] >= mild_min, "mild", "moderate-severe")
    df.loc[df["mmse"].isna(), "stratum"] = np.nan
    return df


def severity(sets: dict[str, pd.DataFrame], participants: pd.DataFrame, class_names: list[str],
             cell: tuple[str, str] = ("P3", "spatial"), mild_min: int = MILD_MIN_MMSE):
    """sets: name -> runs (load_runs rows with preds). Returns (per-subject table, stratum summary, tests,
    per-representation descriptive table for the first set)."""
    cn = class_names.index("CN")
    patients = [i for i, c in enumerate(class_names) if c != "CN"]
    subj, summ, tests = [], [], []
    for name, runs in sets.items():
        long = subject_predictions(runs[(runs.pipeline == cell[0]) & (runs.edge == cell[1])])
        df = stratify(_per_subject(long, cn), participants, mild_min).assign(set=name)
        subj.append(df)
        for c in patients:
            g = df[(df.true_class == c) & df.stratum.notna()]
            for st, h in g.groupby("stratum"):
                summ.append({"set": name, "group": class_names[c], "stratum": st, "n": len(h),
                             "mmse_median": h.mmse.median(), "correct": h.correct.mean(), "p_true": h.p_true.mean(),
                             "pred_cn": h.pred_cn.mean()})
            a, b = g[g.stratum == "mild"].p_true, g[g.stratum == "moderate-severe"].p_true
            if len(a) and len(b):
                u = stats.mannwhitneyu(a, b, alternative="two-sided")
                tests.append({"set": name, "group": class_names[c], "test": "Mann-Whitney p_true mild vs mod-sev",
                              "n": len(g), "stat": float(u.statistic),
                              "effect": float(2 * u.statistic / (len(a) * len(b)) - 1),     # rank-biserial
                              "p": float(u.pvalue)})
            r = stats.spearmanr(g.mmse, g.p_true)
            tests.append({"set": name, "group": class_names[c], "test": "Spearman MMSE vs p_true", "n": len(g),
                          "stat": float(r.statistic), "effect": float(r.statistic), "p": float(r.pvalue)})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"].tolist())

    first = next(iter(sets.values()))
    long = subject_predictions(first)
    rep_rows = []
    for p, lp in long.groupby("pipeline"):
        df = stratify(_per_subject(lp, cn), participants, mild_min)
        for c in patients:
            for st, h in df[(df.true_class == c) & df.stratum.notna()].groupby("stratum"):
                rep_rows.append({"pipeline": p, "group": class_names[c], "stratum": st, "n": len(h),
                                 "correct": h.correct.mean(), "pred_cn": h.pred_cn.mean()})
    return pd.concat(subj, ignore_index=True), pd.DataFrame(summ), tests, pd.DataFrame(rep_rows)


def dropout_sensitivity(main: pd.DataFrame, deep: pd.DataFrame, new: pd.DataFrame,
                        ratio: float = RATIO_TRAIN) -> pd.DataFrame:
    """new: EEGNet (dropout 0.25) runs; deep: the NB06 runs (tag deep_eegnet); main: the GCN grid."""
    cells = main.groupby(["pipeline", "edge"])["macro_f1"].mean()
    best, p5 = cells.idxmax(), cells.loc["P5"].idxmax()
    sel = lambda df, p, e: df[(df.pipeline == p) & (df.edge == e)].set_index("unit")["macro_f1"]
    refs = {"EEGNet (dropout 0.5)": deep[deep.tag == "deep_eegnet"].set_index("unit")["macro_f1"],
            f"GCN {best[0]} x {best[1]}": sel(main, *best), f"GCN P5 x {p5}": sel(main, "P5", p5)}
    a = new.set_index("unit")["macro_f1"]
    rows = []
    for name, b in refs.items():
        d = (a - b.reindex(a.index)).to_numpy()
        rows.append({"model": "EEGNet (dropout 0.25)", "n": len(a), "macro_f1": a.mean(), "sd": a.std(),
                     "balanced_acc": new["balanced_acc"].mean(), "roc_auc": new["roc_auc"].mean(),
                     "reference": name, "reference_f1": b.reindex(a.index).mean(), "delta": d.mean(),
                     "wilcoxon_p": stats.wilcoxon(d, zero_method="zsplit").pvalue, "nb_p": nadeau_bengio(d, ratio)[1]})
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].tolist())
    out["nb_p_holm"] = holm(out["nb_p"].tolist())
    return out
