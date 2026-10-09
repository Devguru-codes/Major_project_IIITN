"""Statistics for the representation × edge factorial (IMPLEMENTATION_PLAN.md §6).

Unit of repeated measurement = one (seed, fold) test split; all 21 cells share the
same 25 units, so every comparison is paired. Fold scores share training data
(they are not independent), hence the Nadeau–Bengio corrected t-test alongside
Wilcoxon. Subject-level bootstrap CIs resample the 88 subjects of each repeat.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import classification_metrics
from .stats_pswe import holm


def load_runs(paths, tag: str | None = None) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if tag and r["tag"] != tag:
                    continue
                rows.append({"tag": r["tag"], "pipeline": r["pipeline"], "edge": r["edge"], "seed": r["seed"],
                             "fold": r["fold"], "config_hash": r["config_hash"],
                             **{f"{k}": v for k, v in r["subject"].items() if k != "confusion"},
                             "confusion": r["subject"]["confusion"],
                             "window_macro_f1": r["window"]["macro_f1"], "seconds": r["seconds"],
                             "epochs_run": r["epochs_run"], "preds": r["preds"]})
    df = pd.DataFrame(rows)
    df["unit"] = df["seed"].astype(str) + "-" + df["fold"].astype(str)
    return df.drop_duplicates(["tag", "pipeline", "edge", "seed", "fold", "config_hash"])


def bootstrap_ci(cell: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Resample subjects within each repeat (pooled over its 5 test folds); average macro-F1 over repeats."""
    rng = np.random.default_rng(seed)
    per_rep = []
    for _, g in cell.groupby("seed"):
        y = np.concatenate([p["y"] for p in g["preds"]])
        prob = np.concatenate([p["prob"] for p in g["preds"]])
        per_rep.append((y, prob))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for y, prob in per_rep:
            i = rng.integers(0, len(y), len(y))
            vals.append(classification_metrics(y[i], prob[i])["macro_f1"])
        draws[b] = np.mean(vals)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cell_table(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    metrics = ["macro_f1", "balanced_acc", "accuracy", "kappa", "roc_auc", "pr_auc", "brier", "ece",
               "sens_0", "spec_0", "sens_1", "spec_1", "sens_2", "spec_2", "window_macro_f1"]
    rows = []
    for (p, e), g in df.groupby(["pipeline", "edge"]):
        row = {"pipeline": p, "edge": e, "n_runs": len(g)}
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std()
        half = stats.t.ppf(0.975, len(g) - 1) * g["macro_f1"].std() / np.sqrt(len(g))
        row["macro_f1_t_ci"] = [row["macro_f1_mean"] - half, row["macro_f1_mean"] + half]
        row["macro_f1_boot_ci"] = list(bootstrap_ci(g, n_boot)) if n_boot else None
        row["seconds_mean"] = g["seconds"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def nadeau_bengio(d: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Corrected resampled t-test (Nadeau & Bengio 2003) on paired score differences."""
    j = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / j + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), j - 1))


def pairwise(wide: pd.DataFrame, test_train_ratio: float) -> pd.DataFrame:
    """wide: index = unit, columns = levels. All pairs: Wilcoxon + NB-corrected t, Holm within each test."""
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        d = (wide[a] - wide[b]).to_numpy()
        w = stats.wilcoxon(wide[a], wide[b], zero_method="zsplit")
        r = stats.rankdata(np.abs(d[d != 0]))
        nz = d[d != 0]
        rbc = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum() if len(r) else 0.0   # matched-pairs rank-biserial, + = a > b
        t, p_nb = nadeau_bengio(d, test_train_ratio)
        rows.append({"a": a, "b": b, "mean_diff": d.mean(), "cohens_d": d.mean() / d.std(ddof=1) if d.std() else 0.0,
                     "wilcoxon_p": w.pvalue, "rank_biserial": rbc, "nb_t": t, "nb_p": p_nb})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].tolist())
        out["nb_p_holm"] = holm(out["nb_p"].tolist())
    return out


def rm_anova(df: pd.DataFrame) -> pd.DataFrame:
    import pingouin as pg

    aov = pg.rm_anova(data=df, dv="macro_f1", within=["pipeline", "edge"], subject="unit", effsize="np2",
                      correction=True)
    return aov


def critical_difference(wide: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test + Nemenyi critical difference (Demšar 2006) over all cells."""
    k, n = wide.shape[1], wide.shape[0]
    fr = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
    ranks = wide.rank(axis=1, ascending=False).mean()
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue), "cd": float(cd),
            "avg_rank": ranks.sort_values().to_dict(), "k": k, "n_units": n}


def analyse_grid(df: pd.DataFrame, n_test_over_train: float = 17.6 / 70.4, n_boot: int = 2000) -> dict:
    df = df[df["tag"] == "main"].copy()
    complete = df.groupby("unit").size()
    n_cells = df.groupby(["pipeline", "edge"]).ngroups
    df = df[df["unit"].isin(complete[complete == n_cells].index)]     # balanced design only
    by_rep = df.groupby(["unit", "pipeline"])["macro_f1"].mean().unstack()
    by_edge = df.groupby(["unit", "edge"])["macro_f1"].mean().unstack()
    cells = df.assign(cell=df["pipeline"] + "×" + df["edge"]).pivot(index="unit", columns="cell", values="macro_f1")
    return {
        "n_units": int(cells.shape[0]), "n_cells": int(n_cells),
        "cells": cell_table(df, n_boot),
        "anova": rm_anova(df),
        "pairwise_representation": pairwise(by_rep, n_test_over_train),
        "pairwise_edge": pairwise(by_edge, n_test_over_train),
        "cd": critical_difference(cells),
        "marginal_representation": by_rep.agg(["mean", "std"]).T,
        "marginal_edge": by_edge.agg(["mean", "std"]).T,
    }


def write_analysis(res: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"grid_{k}.csv", index=k.startswith("marginal"))
    (out / "grid_cd.json").write_text(json.dumps(res["cd"], indent=1))
