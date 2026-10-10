"""Early-stopping sensitivity (added after internal review): the main grid early-stops on a validation set of
~14 subjects with patience 20, and many runs stop within a few epochs. Each variant re-runs the full 7 x 3 grid
on the same 25 selection splits with a different stopping rule; everything else is identical, so every
comparison with the main grid is paired per test split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats_ablation import gcn_vs_classical
from .stats_grid import RATIO_TRAIN, nadeau_bengio, rm_anova
from .stats_pswe import holm

MAIN_RULE = "patience 20 (main)"
ES_RULES = {"es_p50": "patience 50", "es_fixed100": "fixed 100 epochs"}     # run tag -> label


def _cell_means(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["pipeline", "edge"])["macro_f1"].mean()


def es_sensitivity(main: pd.DataFrame, variants: dict[str, pd.DataFrame], classical_train: pd.DataFrame) -> dict:
    rules = {MAIN_RULE: main, **variants}
    ref_cells = _cell_means(main)
    ref_rep = main.groupby(["unit", "pipeline"])["macro_f1"].mean()     # representation score per split
    summary, reps, cells, gvc = [], [], [], []
    for rule, df in rules.items():
        aov = rm_anova(df).set_index("Source")
        cm = _cell_means(df)
        best = cm.idxmax()
        g = gcn_vs_classical(df, classical_train, RATIO_TRAIN).assign(rule=rule)
        gvc.append(g)
        summary.append({
            "rule": rule, "n_runs": len(df), "median_epochs_run": float(df["epochs_run"].median()),
            "median_best_epoch": float(df["best_epoch"].median()),
            "best_cell": f"{best[0]}x{best[1]}", "best_f1": float(cm.max()),
            "p3_spatial_f1": float(cm.get(("P3", "spatial"), np.nan)), "grand_mean_f1": float(cm.mean()),
            "rep_F": aov.loc["pipeline", "F"], "rep_p": aov.loc["pipeline", "p-GG-corr"],
            "rep_np2": aov.loc["pipeline", "np2"], "edge_F": aov.loc["edge", "F"],
            "edge_p": aov.loc["edge", "p-GG-corr"], "inter_p": aov.loc["pipeline * edge", "p-GG-corr"],
            "spearman_vs_main": float(cm.corr(ref_cells.reindex(cm.index), method="spearman")),
            "gcn_vs_classical_n_sig_nb": int((g["nb_p_holm"] < 0.05).sum()),
            "gcn_vs_classical_n_sig_wilcoxon": int((g["wilcoxon_p_holm"] < 0.05).sum()),
        })
        cells.append(cm.rename("macro_f1").reset_index().assign(rule=rule))
        rep = df.groupby(["unit", "pipeline"])["macro_f1"].mean()
        rows = []
        for p in sorted(df["pipeline"].unique()):
            a = rep.xs(p, level="pipeline")
            b = ref_rep.xs(p, level="pipeline").reindex(a.index)
            d = (a - b).to_numpy()
            half = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
            nonzero = np.any(d != 0)
            rows.append({"rule": rule, "pipeline": p, "macro_f1": a.mean(), "ci_lo": a.mean() - half,
                         "ci_hi": a.mean() + half, "delta_vs_main": d.mean(),
                         "wilcoxon_p": stats.wilcoxon(d, zero_method="zsplit").pvalue if nonzero else 1.0,
                         "nb_p": nadeau_bengio(d, RATIO_TRAIN)[1] if nonzero else 1.0})
        r = pd.DataFrame(rows)
        r["wilcoxon_p_holm"] = holm(r["wilcoxon_p"].tolist())
        r["nb_p_holm"] = holm(r["nb_p"].tolist())
        reps.append(r)
    return {"summary": pd.DataFrame(summary), "representations": pd.concat(reps, ignore_index=True),
            "cells": pd.concat(cells, ignore_index=True), "gcn_vs_classical": pd.concat(gvc, ignore_index=True)}


def write_sensitivity(res: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
