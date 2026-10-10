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
                             "epochs_run": r["epochs_run"], "best_epoch": r.get("best_epoch"),
                             "preds": r["preds"]})
    df = pd.DataFrame(rows)
    df["unit"] = df["seed"].astype(str) + "-" + df["fold"].astype(str)
    return df.drop_duplicates(["tag", "pipeline", "edge", "seed", "fold", "config_hash"])


def bootstrap_ci(cell: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Resample subjects within each repeat (pooled over its 5 test folds); average macro-F1 over repeats."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    per_rep = []
    for _, g in cell.groupby("seed"):
        y = np.concatenate([p["y"] for p in g["preds"]])
        pred = np.concatenate([p["prob"] for p in g["preds"]]).argmax(1)
        per_rep.append((y, pred))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for y, pred in per_rep:
            i = rng.integers(0, len(y), len(y))
            vals.append(f1_score(y[i], pred[i], labels=[0, 1, 2], average="macro", zero_division=0))
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


def _orthonormal_contrasts(k: int) -> np.ndarray:
    """k × (k−1) orthonormal contrasts (columns orthogonal to the constant vector)."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, :-1]]))
    return q[:, 1:]


def _gg_epsilon(y: np.ndarray, L: np.ndarray) -> float:
    """Greenhouse–Geisser ε for per-unit vectors y (n × m) and contrast matrix L (m × p)."""
    v = L.T @ np.cov(y, rowvar=False) @ L
    return float(np.trace(v) ** 2 / (L.shape[1] * np.trace(v @ v)))


def rm_anova(df: pd.DataFrame, dv: str = "macro_f1", a: str = "pipeline", b: str = "edge",
             unit: str = "unit") -> pd.DataFrame:
    """Balanced two-way repeated-measures ANOVA (both factors within-unit) with partial η² and
    Greenhouse–Geisser-corrected p-values. Implemented directly from the sums of squares."""
    cube = df.pivot_table(index=unit, columns=[a, b], values=dv, aggfunc="mean")
    la, lb = sorted(df[a].unique()), sorted(df[b].unique())
    cube = cube.reindex(columns=pd.MultiIndex.from_product([la, lb])).dropna()
    n, A, B = len(cube), len(la), len(lb)
    y = cube.to_numpy().reshape(n, A, B)
    m = y.mean()
    ya, yb, yab = y.mean((0, 2)), y.mean((0, 1)), y.mean(0)
    yu, yua, yub = y.mean((1, 2)), y.mean(2), y.mean(1)
    ss = {
        a: (n * B * ((ya - m) ** 2).sum(), B * ((yua - yu[:, None] - ya[None] + m) ** 2).sum(), A - 1),
        b: (n * A * ((yb - m) ** 2).sum(), A * ((yub - yu[:, None] - yb[None] + m) ** 2).sum(), B - 1),
        f"{a} * {b}": (n * ((yab - ya[:, None] - yb[None] + m) ** 2).sum(),
                       ((y - yua[:, :, None] - yub[:, None, :] - yab[None] + ya[None, :, None]
                         + yb[None, None, :] + yu[:, None, None] - m) ** 2).sum(), (A - 1) * (B - 1)),
    }
    La, Lb = _orthonormal_contrasts(A), _orthonormal_contrasts(B)
    eps = {a: _gg_epsilon(yua, La), b: _gg_epsilon(yub, Lb),
           f"{a} * {b}": _gg_epsilon(y.reshape(n, A * B), np.kron(La, Lb))}
    rows = []
    for src, (ss_eff, ss_err, df1) in ss.items():
        df2 = df1 * (n - 1)
        f = (ss_eff / df1) / (ss_err / df2)
        e = min(1.0, eps[src])
        rows.append({"Source": src, "SS": ss_eff, "ddof1": df1, "ddof2": df2, "F": f,
                     "p-unc": stats.f.sf(f, df1, df2), "eps-GG": e, "p-GG-corr": stats.f.sf(f, df1 * e, df2 * e),
                     "np2": ss_eff / (ss_eff + ss_err)})
    return pd.DataFrame(rows)


def critical_difference(wide: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test + Nemenyi critical difference (Demšar 2006) over all cells."""
    k, n = wide.shape[1], wide.shape[0]
    fr = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
    ranks = wide.rank(axis=1, ascending=False).mean()
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue), "cd": float(cd),
            "avg_rank": ranks.sort_values().to_dict(), "k": k, "n_units": n}


RATIO_TRAIN = 17.6 / 56.3      # test / training subjects when models train on the train split only (GCN)
RATIO_TRAINVAL = 17.6 / 70.4   # ... and when they train on train + validation (refit GCN, classical)


def analyse_grid(df: pd.DataFrame, n_test_over_train: float = RATIO_TRAIN, n_boot: int = 2000) -> dict:
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
