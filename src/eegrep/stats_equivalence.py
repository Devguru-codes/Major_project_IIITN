"""Equivalence tests and minimum detectable effects for the paper's null claims (added after internal review).

A non-significant difference is not evidence of no difference. For every paired comparison the paper reads as
"no advantage" we report two one-sided tests (TOST) against a margin of +/-0.05 macro-F1 (about one more
correctly classified subject in a test fold of 17-18), the 90% confidence interval, and the minimum effect
detectable with 80% power. All use the Nadeau-Bengio corrected standard error, so they are conservative: a
comparison is called equivalent only if the whole corrected 90% interval lies inside the margin.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats_grid import RATIO_TRAIN, RATIO_TRAINVAL, load_runs
from .stats_pswe import holm

MARGIN = 0.05


def paired(a: pd.Series, b: pd.Series, ratio: float, margin: float = MARGIN) -> dict:
    """a, b: macro-F1 indexed by split unit ("seed-fold"). Difference a - b."""
    d = (a - b.reindex(a.index)).dropna().to_numpy()
    n, m = len(d), float(d.mean())
    se = float(np.sqrt((1.0 / n + ratio) * d.var(ddof=1))) if n > 1 else float("nan")
    df = n - 1
    if not se > 0:                                        # identical runs
        return {"n": n, "delta": m, "se_nb": 0.0, "ci90_lo": m, "ci90_hi": m, "p_tost": 0.0, "mde80": 0.0}
    q = stats.t.ppf(0.95, df)
    p_lower = stats.t.sf((m + margin) / se, df)           # H0: delta <= -margin
    p_upper = stats.t.cdf((m - margin) / se, df)          # H0: delta >= +margin
    return {"n": n, "delta": m, "se_nb": se, "ci90_lo": m - q * se, "ci90_hi": m + q * se,
            "p_tost": float(max(p_lower, p_upper)),
            "mde80": float((stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se)}


def _cell(df: pd.DataFrame, p: str, e: str) -> pd.Series:
    return df[(df.pipeline == p) & (df.edge == e)].set_index("unit")["macro_f1"]


def _units(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(unit=df["seed"].astype(str) + "-" + df["fold"].astype(str))


def build(reports: str | Path, margin: float = MARGIN) -> pd.DataFrame:
    rep = Path(reports)
    g = lambda pat: sorted(glob.glob(str(rep / pat)))
    main = load_runs(g("grid/runs_s*.jsonl"), tag="main")
    rows = []

    def add(family, comparison, setting, a, b, ratio):
        rows.append({"family": family, "comparison": comparison, "setting": setting, **paired(a, b, ratio, margin)})

    # 1. GCN vs its best non-graph twin, both training regimes (as in Table S1)
    regimes = [("train only", main, _units(pd.read_csv(rep / "confirm" / "classical_train.csv")), RATIO_TRAIN)]
    if g("confirm/refit_runs_s*.jsonl"):
        regimes.append(("train + val", load_runs(g("confirm/refit_runs_s*.jsonl"), tag="refit"),
                        _units(pd.read_csv(rep / "nb03c_classical.csv")), RATIO_TRAINVAL))
    for regime, gcn, cls, ratio in regimes:
        for p in sorted(set(gcn.pipeline) & set(cls.pipeline)):
            e = gcn[gcn.pipeline == p].groupby("edge")["macro_f1"].mean().idxmax()
            c = cls[cls.pipeline == p]
            m = c.groupby("model")["macro_f1"].mean().idxmax()
            add("GCN vs non-graph twin", f"{p}: GCN ({e}) - {m}", regime, _cell(gcn, p, e),
                c[c.model == m].set_index("unit")["macro_f1"], ratio)

    # 2. graph construction: marginal (averaged over representations) per split, pairwise
    sets = [("selection", main)]
    if g("confirm/confirm_runs_s*.jsonl"):
        sets.append(("confirmation", load_runs(g("confirm/confirm_runs_s*.jsonl"), tag="confirm")))
    for name, df in sets:
        marg = df.groupby(["unit", "edge"])["macro_f1"].mean().unstack()
        for a, b in (("spatial", "functional"), ("spatial", "hybrid"), ("functional", "hybrid")):
            add("graph construction", f"{a} - {b}", name, marg[a], marg[b], RATIO_TRAIN)

    # 3. ablations on the confirmation folds (paired with the same cell's confirmation-grid run)
    if len(sets) > 1 and g("confirm/ablations_confirm_*.jsonl"):
        conf = sets[1][1]
        abl = load_runs(g("confirm/ablations_confirm_*.jsonl"))
        for tag, d in abl[~abl.tag.str.startswith(("perm", "A10"))].groupby("tag"):
            name, cell = tag.split("@")
            p, e = cell.split("x")
            add("ablation", f"{name} ({p} x {e})", "confirmation", d.set_index("unit")["macro_f1"],
                _cell(conf, p, e), RATIO_TRAIN)

    # 4. deep-learning baselines vs the best GCN cell and the raw-window GCN
    if g("deep/deep_runs_s*.jsonl"):
        deep = load_runs(g("deep/deep_runs_s*.jsonl"))
        cells = main.groupby(["pipeline", "edge"])["macro_f1"].mean()
        best, p5 = cells.idxmax(), cells.loc["P5"].idxmax()
        for tag, d in deep.groupby("tag"):
            a = d.set_index("unit")["macro_f1"]
            add("deep baseline", f"{tag.replace('deep_', '')} - GCN {best[0]} x {best[1]}", "selection", a,
                _cell(main, *best), RATIO_TRAIN)
            add("deep baseline", f"{tag.replace('deep_', '')} - GCN P5 x {p5}", "selection", a,
                _cell(main, "P5", p5), RATIO_TRAIN)

    # 5. stopping rule: per-representation score (averaged over graph types) vs the main grid
    if g("es_sensitivity/es_runs_s*.jsonl"):
        es = load_runs(g("es_sensitivity/es_runs_s*.jsonl"))
        ref = main.groupby(["unit", "pipeline"])["macro_f1"].mean().unstack()
        for tag, d in es.groupby("tag"):
            v = d.groupby(["unit", "pipeline"])["macro_f1"].mean().unstack()
            for p in v.columns:
                add("stopping rule", f"{tag} - main ({p})", "selection", v[p], ref[p], RATIO_TRAIN)

    # 6. learned adjacency (trainable graph initialised at the spatial graph) vs the fixed spatial graph
    if g("learned/learned_runs_s*.jsonl"):
        lr = load_runs(g("learned/learned_runs_s*.jsonl"))
        for tag, ref, name in (("learned_adj", main, "selection"),
                               ("learned_adj_confirm", sets[1][1] if len(sets) > 1 else None, "confirmation")):
            d = lr[lr.tag == tag]
            if ref is None or d.empty:
                continue
            for p in sorted(d.pipeline.unique()):
                add("learned adjacency", f"{p}: learned - spatial", name, _cell(d, p, "spatial"),
                    _cell(ref, p, "spatial"), RATIO_TRAIN)

    out = pd.DataFrame(rows)
    out["p_tost_holm"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "p_tost_holm"] = holm(out.loc[idx, "p_tost"].tolist())
    out["equivalent"] = out["p_tost_holm"] < 0.05
    out["margin"] = margin
    return out


def summarise(tab: pd.DataFrame) -> pd.DataFrame:
    return tab.groupby(["family", "setting"]).agg(n_comparisons=("comparison", "size"),
                                                  n_equivalent=("equivalent", "sum"),
                                                  median_mde80=("mde80", "median"),
                                                  max_abs_delta=("delta", lambda s: s.abs().max())).reset_index()
