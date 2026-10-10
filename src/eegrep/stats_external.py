"""External validation on ds004584 (Parkinson's disease vs controls; IMPLEMENTATION_PLAN.md §5, RQ5).

Different site, device, task (eyes open), mains frequency and disease, so classifiers are not transferred;
the protocol is. Questions fixed before the run (IMPLEMENTATION_PLAN.md log, 2026-10-10):
  1. Does the ranking of the 21 representation x graph cells (and of the 7 representations) found on ds004504
     transfer to ds004584?  Spearman/Kendall between cell means.
  2. How does the configuration selected on ds004504 (P3 x spatial) score on ds004584, with no re-selection?
  3. Do graph models beat their non-graph twins (same training subjects), and does leakage inflate here too?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats_ablation import baseline_table, gcn_vs_classical, leakage_table, permutation_test
from .stats_grid import RATIO_TRAIN, analyse_grid


def transfer(src_cells: pd.DataFrame, ext_cells: pd.DataFrame) -> dict:
    a = src_cells.set_index(["pipeline", "edge"])["macro_f1_mean"]
    b = ext_cells.set_index(["pipeline", "edge"])["macro_f1_mean"].reindex(a.index)
    ra = a.groupby(level="pipeline").mean()
    rb = b.groupby(level="pipeline").mean().reindex(ra.index)
    out = {"n_cells": int(len(a)), "n_representations": int(len(ra))}
    for name, x, y in (("cells", a, b), ("representations", ra, rb)):
        s, k = stats.spearmanr(x, y), stats.kendalltau(x, y)
        out[name] = {"spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue),
                     "kendall_tau": float(k.statistic), "kendall_p": float(k.pvalue)}
    out["representation_means"] = {"source": ra.round(4).to_dict(), "external": rb.round(4).to_dict()}
    return out


def external_analysis(src_cells: pd.DataFrame, ext_main: pd.DataFrame, classical_train: pd.DataFrame,
                      classical_trainval: pd.DataFrame, demographics: pd.DataFrame, abl: pd.DataFrame,
                      selected: tuple[str, str] = ("P3", "spatial"), n_boot: int = 2000) -> dict:
    grid = analyse_grid(ext_main, n_boot=n_boot)
    cells = grid["cells"].reset_index(drop=True)
    pos = cells.index[(cells.pipeline == selected[0]) & (cells.edge == selected[1])][0]
    sel = cells.loc[pos]
    demo = demographics.groupby("model")["macro_f1"].mean()
    res = {
        "grid": grid,
        "transfer": transfer(src_cells, cells),
        "selected": {"cell": f"{selected[0]}x{selected[1]}", "rank": int(pos) + 1, "macro_f1": float(sel.macro_f1_mean),
                     "macro_f1_sd": float(sel.macro_f1_sd), "boot_ci": sel.macro_f1_boot_ci,
                     "balanced_acc": float(sel.balanced_acc_mean), "roc_auc": float(sel.roc_auc_mean),
                     "kappa": float(sel.kappa_mean)},
        "best_cell": {"cell": f"{cells.loc[0].pipeline}x{cells.loc[0].edge}", "macro_f1": float(cells.loc[0].macro_f1_mean)},
        "demographics_f1": float(demo.get("demographics_lr", np.nan)),
        "chance_f1": float(demo.get("chance_stratified", np.nan)),
        "baselines": baseline_table(ext_main, classical_trainval, demographics),
        "gcn_vs_classical_train": gcn_vs_classical(ext_main, classical_train, RATIO_TRAIN),
        "leakage": leakage_table(ext_main, abl),
        "permutation": permutation_test(ext_main, abl),
    }
    return res


def write_external(res: dict, out_dir: str | Path) -> dict:
    """Tables to CSV; returns the JSON-able summary."""
    import json

    from .stats_grid import write_analysis

    out = Path(out_dir)
    write_analysis(res["grid"], out)
    for k in ("baselines", "gcn_vs_classical_train", "leakage"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    perm = {k: {kk: vv for kk, vv in v.items() if kk != "null"} for k, v in res["permutation"].items()}
    summary = {"transfer": res["transfer"], "selected": res["selected"], "best_cell": res["best_cell"],
               "demographics_f1": res["demographics_f1"], "chance_f1": res["chance_f1"],
               "anova": res["grid"]["anova"].to_dict("records"),
               "marginal_representation": res["grid"]["marginal_representation"]["mean"].round(4).to_dict(),
               "marginal_edge": res["grid"]["marginal_edge"]["mean"].round(4).to_dict(),
               "leakage": res["leakage"].round(4).to_dict("records"), "permutation": perm,
               "gcn_vs_classical_train": res["gcn_vs_classical_train"].round(4).to_dict("records")}
    (out / "external_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "permutation.json").write_text(json.dumps(res["permutation"], indent=1))
    return summary
