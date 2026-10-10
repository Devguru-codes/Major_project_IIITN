"""Primary inference on the factorial: a linear mixed model on subject-level outcomes (added after internal review).

The split-level repeated-measures ANOVA treats the 25 cross-validation splits as independent units although they
share subjects and training data. Here the unit is the held-out subject: every subject is scored once per repeat
by every cell's model, and the outcome is the probability that model assigns to the subject's true class.

    p_true ~ representation * graph (sum-to-zero) + true class
             + (1 | subject) + (1 | split) + (1 | model = cell x split)

Crossed random intercepts absorb subject difficulty, the shared train/test partition, and the quality of each
trained network (the subjects scored by one model share it; this is the error term for cell comparisons).
Fixed-effect terms are tested with Wald chi-square tests; marginal means average over graph types (or
representations) and the observed class mix; pairwise representation contrasts are Holm-corrected.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats_pswe import holm

TERMS = {"C(pipeline, Sum)": "representation", "C(edge, Sum)": "graph",
         "C(pipeline, Sum):C(edge, Sum)": "interaction", "C(true_class)": "true class"}


def subject_long(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, held-out subject) from the per-subject predictions stored with every run."""
    rows = []
    for r in runs.itertuples(index=False):
        prob = np.asarray(r.preds["prob"])
        y = np.asarray(r.preds["y"])
        unit = f"{r.seed}-{r.fold}"
        for s, yy, pp in zip(r.preds["subjects"], y, prob):
            rows.append((s, int(yy), float(pp[yy]), int(pp.argmax() == yy), r.pipeline, r.edge, unit,
                         f"{r.pipeline}|{r.edge}|{unit}"))
    return pd.DataFrame(rows, columns=["subject", "true_class", "p_true", "correct", "pipeline", "edge", "unit",
                                       "model"])


RHS = "C(pipeline, Sum) * C(edge, Sum) + C(true_class)"


def fit_mixed(long: pd.DataFrame, outcome: str = "p_true"):
    import statsmodels.formula.api as smf

    d = long.assign(one=1)
    md = smf.mixedlm(f"{outcome} ~ {RHS}", d, groups="one",
                     re_formula="0", vc_formula={"subject": "0 + C(subject)", "split": "0 + C(unit)",
                                                 "model": "0 + C(model)"})
    return md.fit(reml=True, method=["lbfgs"])


def _design(fit, data: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    """Fixed-effects design rows for `frame`, using the coding learnt from the fitted data (same RHS formula)."""
    from patsy import build_design_matrices, dmatrix

    info = dmatrix(RHS, data).design_info
    assert list(info.column_names) == list(fit.fe_params.index), "design columns differ from fitted effects"
    return np.asarray(build_design_matrices([info], frame)[0])


def analyse_mixed(long: pd.DataFrame, outcome: str = "p_true") -> dict:
    t0 = time.time()
    fit = fit_mixed(long, outcome)
    k = fit.k_fe
    beta, cov = fit.fe_params.to_numpy(), fit.cov_params().iloc[:k, :k].to_numpy()
    # Wald chi-square per term (sum-to-zero coding: main effects are averaged over the other factor)
    names = list(fit.fe_params.index)
    terms = []
    for term, label in TERMS.items():
        idx = [i for i, n in enumerate(names) if n.startswith(term) and (":" in n) == (":" in term)]
        b, v = beta[idx], cov[np.ix_(idx, idx)]
        chi2 = float(b @ np.linalg.solve(v, b))
        terms.append({"term": label, "df": len(idx), "chi2": chi2, "p": float(stats.chi2.sf(chi2, len(idx)))})
    # marginal means over the full grid x observed class mix
    classes = long.groupby("subject")["true_class"].first().value_counts(normalize=True).sort_index()
    pipes, edges = sorted(long.pipeline.unique()), sorted(long.edge.unique())
    grid = pd.DataFrame([(p, e, c) for p in pipes for e in edges for c in classes.index],
                        columns=["pipeline", "edge", "true_class"])
    grid[outcome] = 0.0
    X = _design(fit, long, grid)
    w = grid["true_class"].map(classes).to_numpy()

    def marginal(mask):
        L = (X[mask] * w[mask, None]).sum(0) / w[mask].sum()
        return L, float(L @ beta), float(np.sqrt(L @ cov @ L))

    margins = []
    for factor, levels in (("pipeline", pipes), ("edge", edges)):
        for lev in levels:
            _, m, se = marginal((grid[factor] == lev).to_numpy())
            margins.append({"factor": factor, "level": lev, "mean": m, "se": se,
                            "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se})
    pairs = []
    for a, b in itertools.combinations(pipes, 2):
        La, ma, _ = marginal((grid.pipeline == a).to_numpy())
        Lb, mb, _ = marginal((grid.pipeline == b).to_numpy())
        L = La - Lb
        se = float(np.sqrt(L @ cov @ L))
        z = (ma - mb) / se
        pairs.append({"a": a, "b": b, "diff": ma - mb, "se": se, "z": z, "p": float(2 * stats.norm.sf(abs(z)))})
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = holm(pairs["p"].tolist())
    vc = dict(zip(fit.model.exog_vc.names, np.asarray(fit.vcomp, dtype=float)))
    var = {**vc, "residual": float(fit.scale)}
    total = sum(var.values())
    return {"outcome": outcome, "n_rows": int(len(long)), "n_subjects": int(long.subject.nunique()),
            "n_models": int(long.model.nunique()), "converged": bool(fit.converged),
            "seconds": time.time() - t0, "terms": pd.DataFrame(terms), "marginal": pd.DataFrame(margins),
            "pairwise_representation": pairs,
            "variance_components": {k_: {"var": v, "share": v / total} for k_, v in var.items()},
            "fixed_effects": fit.fe_params.round(6).to_dict()}


def write_mixed(res: dict, out_dir: str | Path, prefix: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("terms", "marginal", "pairwise_representation"):
        res[k].to_csv(out / f"{prefix}_{k}.csv", index=False)
    summary = {k: v for k, v in res.items() if not isinstance(v, pd.DataFrame)}
    summary["terms"] = res["terms"].to_dict("records")
    summary["n_pairs_sig_holm"] = int((res["pairwise_representation"]["p_holm"] < 0.05).sum())
    (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
