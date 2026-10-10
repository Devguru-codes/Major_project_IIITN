"""Primary inference on the factorial: a by-subject linear mixed model (added after internal review).

The split-level repeated-measures ANOVA treats the 25 cross-validation splits as the units of inference. Here the
unit is the held-out subject, the population a diagnostic claim is about. Every subject is scored once per repeat
by every cell's model; the outcome is the probability that model assigns to the subject's true class, averaged
over the five repeats (each from a different partition), giving one value per subject x cell:

    p_true ~ representation * graph (sum-to-zero) + true class
             + (1 | subject) + (1 | subject:representation) + (1 | subject:graph)

The subject-by-factor random effects are essential: a subject's advantage under one representation recurs in every
repeat and graph type, so testing representation against row-level residuals would treat ~9,000 correlated scores
as independent (pseudo-replication; a first version without them gave implausibly small p-values and was
discarded, see IMPLEMENTATION_PLAN.md). With them, each term is tested against its own subject-level stratum,
as in a by-subject repeated-measures ANOVA, to which this model reduces for balanced data (tested against
statsmodels AnovaRM). Wald F tests use containment denominator df; marginal means average over the other factor
and the observed class mix; pairwise representation contrasts are Holm-corrected.
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
RHS = "C(pipeline, Sum) * C(edge, Sum) + C(true_class)"


def subject_long(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, held-out subject) from the per-subject predictions stored with every run."""
    rows = []
    for r in runs.itertuples(index=False):
        prob = np.asarray(r.preds["prob"])
        y = np.asarray(r.preds["y"])
        unit = f"{r.seed}-{r.fold}"
        for s, yy, pp in zip(r.preds["subjects"], y, prob):
            rows.append((s, int(yy), float(pp[yy]), int(pp.argmax() == yy), r.pipeline, r.edge, unit))
    return pd.DataFrame(rows, columns=["subject", "true_class", "p_true", "correct", "pipeline", "edge", "unit"])


def subject_cell_means(long: pd.DataFrame, outcome: str = "p_true") -> pd.DataFrame:
    """Average each subject's outcome over the repeats within each cell (one row per subject x cell)."""
    out = long.groupby(["subject", "true_class", "pipeline", "edge"], as_index=False).agg(
        **{outcome: (outcome, "mean"), "n_repeats": (outcome, "size")})
    n_cells = long.groupby(["pipeline", "edge"]).ngroups
    complete = out.groupby("subject")["pipeline"].transform("size") == n_cells
    assert complete.all(), "every subject must be tested in every cell (balanced design)"
    return out


def fit_mixed(cm: pd.DataFrame, outcome: str = "p_true", rhs: str = RHS):
    import statsmodels.formula.api as smf

    md = smf.mixedlm(f"{outcome} ~ {rhs}", cm, groups="subject", re_formula="1",
                     vc_formula={"representation": "0 + C(pipeline)", "graph": "0 + C(edge)"})
    return md.fit(reml=True, method=["lbfgs", "powell"])     # Powell only if L-BFGS does not converge


def _design(fit, data: pd.DataFrame, frame: pd.DataFrame, rhs: str) -> np.ndarray:
    """Fixed-effects design rows for `frame`, using the coding learnt from the fitted data (same RHS formula)."""
    from patsy import build_design_matrices, dmatrix

    info = dmatrix(rhs, data).design_info
    assert list(info.column_names) == list(fit.fe_params.index), "design columns differ from fitted effects"
    return np.asarray(build_design_matrices([info], frame)[0])


def analyse_mixed(long: pd.DataFrame, outcome: str = "p_true", with_class: bool = True) -> dict:
    t0 = time.time()
    rhs = RHS if with_class else RHS.replace(" + C(true_class)", "")
    cm = subject_cell_means(long, outcome)
    fit = fit_mixed(cm, outcome, rhs)
    k = fit.k_fe
    beta, cov = fit.fe_params.to_numpy(), fit.cov_params().iloc[:k, :k].to_numpy()
    names = list(fit.fe_params.index)
    n_subj, n_rep, n_edge = cm.subject.nunique(), cm.pipeline.nunique(), cm.edge.nunique()
    n_between = n_subj - (cm.true_class.nunique() if with_class else 1)
    # containment df: each within-subject term is tested in its subject-level stratum
    den = {"representation": n_between * (n_rep - 1), "graph": n_between * (n_edge - 1),
           "interaction": n_between * (n_rep - 1) * (n_edge - 1), "true class": n_between}
    terms = []
    for term, label in TERMS.items():
        idx = [i for i, n in enumerate(names) if ":".join(part.split("[")[0] for part in n.split(":")) == term]
        if not idx:
            continue
        b, v = beta[idx], cov[np.ix_(idx, idx)]
        chi2 = float(b @ np.linalg.solve(v, b))
        f = chi2 / len(idx)
        terms.append({"term": label, "df": len(idx), "chi2": chi2, "p_chi2": float(stats.chi2.sf(chi2, len(idx))),
                      "F": f, "den_df": den[label], "p": float(stats.f.sf(f, len(idx), den[label]))})
    # marginal means over the full grid x observed class mix
    classes = cm.groupby("subject")["true_class"].first().value_counts(normalize=True).sort_index()
    pipes, edges = sorted(cm.pipeline.unique()), sorted(cm.edge.unique())
    grid = pd.DataFrame([(p, e, c) for p in pipes for e in edges for c in classes.index],
                        columns=["pipeline", "edge", "true_class"])
    X = _design(fit, cm, grid, rhs)
    w = grid["true_class"].map(classes).to_numpy()

    def marginal(mask):
        L = (X[mask] * w[mask, None]).sum(0) / w[mask].sum()
        return L, float(L @ beta), float(np.sqrt(L @ cov @ L))

    margins = []
    for factor, levels in (("pipeline", pipes), ("edge", edges)):
        q = stats.t.ppf(0.975, den["representation" if factor == "pipeline" else "graph"])
        for lev in levels:
            _, m, se = marginal((grid[factor] == lev).to_numpy())
            margins.append({"factor": factor, "level": lev, "mean": m, "se": se,
                            "ci_lo": m - q * se, "ci_hi": m + q * se})
    pairs = []
    for a, b in itertools.combinations(pipes, 2):
        La, ma, _ = marginal((grid.pipeline == a).to_numpy())
        Lb, mb, _ = marginal((grid.pipeline == b).to_numpy())
        L = La - Lb
        se = float(np.sqrt(L @ cov @ L))
        tval = (ma - mb) / se
        pairs.append({"a": a, "b": b, "diff": ma - mb, "se": se, "t": tval,
                      "p": float(2 * stats.t.sf(abs(tval), den["representation"]))})
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = holm(pairs["p"].tolist())
    var = {"subject": float(np.asarray(fit.cov_re)[0, 0]),
           **{f"subject x {n}": float(v) for n, v in zip(fit.model.exog_vc.names, np.asarray(fit.vcomp))},
           "residual (subject x cell)": float(fit.scale)}
    total = sum(var.values())
    return {"outcome": outcome, "n_rows": int(len(cm)), "n_scores": int(len(long)), "n_subjects": int(n_subj),
            "converged": bool(fit.converged), "seconds": time.time() - t0, "terms": pd.DataFrame(terms),
            "marginal": pd.DataFrame(margins), "pairwise_representation": pairs,
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
