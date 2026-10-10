"""Ablation, leakage, permutation and baseline tables (IMPLEMENTATION_PLAN.md §5–§6).

Every ablation run uses the same frozen (seed, fold) splits as its main-grid cell, so
differences are paired per test split. A10 (window-level splitting) uses different
splits by construction and is summarised unpaired.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats_grid import RATIO_TRAIN, nadeau_bengio
from .stats_pswe import holm


def split_tag(tag: str) -> tuple[str, str, str]:
    name, cell = tag.split("@")
    p, e = cell.split("x")
    return name, p, e


def ablation_table(main: pd.DataFrame, abl: pd.DataFrame, test_train_ratio: float = RATIO_TRAIN) -> pd.DataFrame:
    rows = []
    for tag, g in abl[~abl["tag"].str.startswith(("perm", "A10"))].groupby("tag"):
        name, p, e = split_tag(tag)
        base = main[(main.pipeline == p) & (main.edge == e)].set_index("unit")["macro_f1"]
        ab = g.set_index("unit")["macro_f1"]
        units = base.index.intersection(ab.index)
        d = (ab.loc[units] - base.loc[units]).to_numpy()
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        w = stats.wilcoxon(d, zero_method="zsplit") if np.any(d != 0) else None
        t, p_nb = nadeau_bengio(d, test_train_ratio)
        rows.append({"cell": f"{p}x{e}", "ablation": name, "n": len(d), "reference_f1": base.loc[units].mean(),
                     "ablated_f1": ab.loc[units].mean(), "delta": d.mean(), "delta_ci_lo": d.mean() - half,
                     "delta_ci_hi": d.mean() + half, "cohens_d": d.mean() / d.std(ddof=1) if d.std() else 0.0,
                     "wilcoxon_p": w.pvalue if w else 1.0, "nb_t": t, "nb_p": p_nb})
    out = pd.DataFrame(rows)
    for cell, idx in out.groupby("cell").groups.items():          # Holm within each cell's ablation family
        out.loc[idx, "wilcoxon_p_holm"] = holm(out.loc[idx, "wilcoxon_p"].tolist())
        out.loc[idx, "nb_p_holm"] = holm(out.loc[idx, "nb_p"].tolist())
    return out.sort_values(["cell", "delta"])


def leakage_table(main: pd.DataFrame, abl: pd.DataFrame) -> pd.DataFrame:
    """A10: identical pipeline/model/data; only the split protocol differs."""
    rows = []
    for tag, g in abl[abl["tag"].str.startswith("A10")].groupby("tag"):
        _, p, e = split_tag(tag)
        ref = main[(main.pipeline == p) & (main.edge == e)]
        for protocol, df in (("subject-level (correct)", ref), ("window-level (leaky)", g)):
            rows.append({"cell": f"{p}x{e}", "protocol": protocol,
                         "subject_macro_f1": df["macro_f1"].mean(), "subject_macro_f1_sd": df["macro_f1"].std(),
                         "window_macro_f1": df["window_macro_f1"].mean(),
                         "window_macro_f1_sd": df["window_macro_f1"].std(),
                         "subject_accuracy": df["accuracy"].mean()})
    out = pd.DataFrame(rows)
    if len(out):
        piv = out.set_index(["cell", "protocol"])
        for cell in out["cell"].unique():
            out.loc[out.cell == cell, "inflation_window_vs_correct"] = (
                piv.loc[(cell, "window-level (leaky)"), "window_macro_f1"]
                - piv.loc[(cell, "subject-level (correct)"), "subject_macro_f1"])
    return out


def permutation_test(main: pd.DataFrame, abl: pd.DataFrame) -> dict:
    out = {}
    for tag, g in abl[abl["tag"].str.startswith("perm")].groupby("tag"):
        _, p, e = split_tag(tag)
        per = g.groupby("seed")["macro_f1"].agg(["mean", "size"])
        null = per.loc[per["size"] == per["size"].max(), "mean"].to_numpy()
        ref = main[(main.pipeline == p) & (main.edge == e) & (main.seed == 0)]["macro_f1"].mean()
        out[f"{p}x{e}"] = {"n_perm": int(len(null)), "observed_seed0_macro_f1": float(ref),
                           "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                           "null_max": float(null.max()), "p": float((1 + (null >= ref).sum()) / (1 + len(null))),
                           "null": null.round(4).tolist()}
    return out


def baseline_table(main: pd.DataFrame, classical: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """Table 4: best GCN cell per representation vs its non-graph twins and the confound floor."""
    rows = []
    for p, g in main.groupby("pipeline"):
        best = g.groupby("edge")["macro_f1"].mean().idxmax()
        v = g[g.edge == best]["macro_f1"]
        rows.append({"representation": p, "model": f"GCN ({best})", "macro_f1": v.mean(), "sd": v.std(), "n": len(v)})
    for (p, m), g in classical.groupby(["pipeline", "model"]):
        rows.append({"representation": p, "model": m, "macro_f1": g["macro_f1"].mean(), "sd": g["macro_f1"].std(),
                     "n": len(g)})
    for m, g in demographics.groupby("model"):
        rows.append({"representation": "age+sex", "model": m, "macro_f1": g["macro_f1"].mean(),
                     "sd": g["macro_f1"].std(), "n": len(g)})
    return pd.DataFrame(rows).sort_values(["representation", "macro_f1"], ascending=[True, False])


def gcn_vs_classical(main: pd.DataFrame, classical: pd.DataFrame, test_train_ratio: float = RATIO_TRAIN) -> pd.DataFrame:
    """Paired (seed, fold) comparison of each representation's best GCN cell vs its best classical twin."""
    classical = classical.assign(unit=classical["seed"].astype(str) + "-" + classical["fold"].astype(str))
    rows = []
    for p, g in main.groupby("pipeline"):
        if p not in set(classical.pipeline):
            continue
        best_e = g.groupby("edge")["macro_f1"].mean().idxmax()
        c = classical[classical.pipeline == p]
        best_m = c.groupby("model")["macro_f1"].mean().idxmax()
        a = g[g.edge == best_e].set_index("unit")["macro_f1"]
        b = c[c.model == best_m].set_index("unit")["macro_f1"]
        d = (a - b.loc[a.index]).to_numpy()
        t, p_nb = nadeau_bengio(d, test_train_ratio)
        rows.append({"representation": p, "gcn": best_e, "classical": best_m, "gcn_f1": a.mean(),
                     "classical_f1": b.mean(), "delta": d.mean(),
                     "wilcoxon_p": stats.wilcoxon(d, zero_method="zsplit").pvalue, "nb_p": p_nb})
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].tolist())
    out["nb_p_holm"] = holm(out["nb_p"].tolist())
    return out


def write_all(main, abl, classical, demographics, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"ablations": ablation_table(main, abl), "leakage": leakage_table(main, abl),
           "baselines": baseline_table(main, classical, demographics),
           "gcn_vs_classical": gcn_vs_classical(main, classical)}
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    res["permutation"] = permutation_test(main, abl)
    (out / "permutation.json").write_text(json.dumps(res["permutation"], indent=1))
    return res
