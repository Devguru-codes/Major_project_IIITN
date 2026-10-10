"""By-subject mixed model: matches the by-subject RM-ANOVA, recovers a known effect, and is calibrated under the
null even when subjects differ in how well each representation suits them (the case that breaks a row-level model)."""
import numpy as np
import pandas as pd
import pytest

from eegrep.stats_mixed import analyse_mixed, subject_cell_means, subject_long


def _simulate(rng, rep_effect, n_subj=30, reps=("P1", "P2", "P3"), edges=("hybrid", "spatial"), repeats=5, folds=3,
              sd_subj=0.10, sd_subj_rep=0.06, sd_subj_edge=0.02, sd_noise=0.08):
    subj = [f"sub-{i:03d}" for i in range(n_subj)]
    cls = {s: i % 3 for i, s in enumerate(subj)}
    u = {s: rng.normal(0, sd_subj) for s in subj}
    ur = {(s, p): rng.normal(0, sd_subj_rep) for s in subj for p in reps}
    ue = {(s, e): rng.normal(0, sd_subj_edge) for s in subj for e in edges}
    rows = []
    for r in range(repeats):
        order = rng.permutation(subj)
        for k in range(folds):
            unit, u_split = f"{r}-{k}", rng.normal(0, 0.03)
            for p in reps:
                for e in edges:
                    for s in order[k::folds]:
                        y = 0.45 + rep_effect[p] + u[s] + ur[(s, p)] + ue[(s, e)] + u_split + rng.normal(0, sd_noise)
                        rows.append((s, cls[s], y, int(y > 0.5), p, e, unit))
    return pd.DataFrame(rows, columns=["subject", "true_class", "p_true", "correct", "pipeline", "edge", "unit"])


def test_matches_by_subject_rm_anova(rng):
    from statsmodels.stats.anova import AnovaRM

    long = _simulate(rng, {"P1": 0.0, "P2": -0.03, "P3": 0.02})
    t = analyse_mixed(long, with_class=False)["terms"].set_index("term")
    cm = subject_cell_means(long)
    ref = AnovaRM(cm, "p_true", "subject", within=["pipeline", "edge"]).fit().anova_table
    for ours, theirs in (("representation", "pipeline"), ("graph", "edge"), ("interaction", "pipeline:edge")):
        assert t.loc[ours, "F"] == pytest.approx(ref.loc[theirs, "F Value"], rel=0.02)
        assert t.loc[ours, "den_df"] == ref.loc[theirs, "Den DF"]


def test_recovers_representation_effect(rng):
    res = analyse_mixed(_simulate(rng, {"P1": 0.0, "P2": -0.08, "P3": 0.06}))
    t = res["terms"].set_index("term")
    assert t.loc["representation", "p"] < 1e-3 and t.loc["representation", "df"] == 2
    assert t.loc["graph", "df"] == 1 and t.loc["interaction", "df"] == 2 and t.loc["true class", "df"] == 2
    m = res["marginal"].query("factor == 'pipeline'").set_index("level")["mean"]
    assert m["P3"] > m["P1"] > m["P2"] and m["P3"] - m["P2"] == pytest.approx(0.14, abs=0.04)
    pw = res["pairwise_representation"].set_index(["a", "b"])
    assert pw.loc[("P2", "P3"), "p_holm"] < 0.01
    vc = res["variance_components"]
    assert vc["subject x representation"]["var"] > vc["subject x graph"]["var"]


def test_null_false_positive_rate_is_nominal():
    """30 null datasets with real subject x representation variation: rejection rate ~5%, p roughly uniform."""
    p = [analyse_mixed(_simulate(np.random.default_rng(100 + i), {"P1": 0.0, "P2": 0.0, "P3": 0.0}))["terms"]
         .set_index("term").loc["representation", "p"] for i in range(30)]
    assert np.mean(np.array(p) < 0.05) <= 0.15
    assert np.mean(p) > 0.3


def test_subject_long_and_cell_means():
    runs = pd.DataFrame([{"pipeline": "P1", "edge": "spatial", "seed": s, "fold": 0,
                          "preds": {"subjects": ["a", "b"], "y": [0, 2], "prob": [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]]}}
                         for s in (0, 1)])
    long = subject_long(runs)
    assert long["p_true"].tolist() == [0.7, 0.2, 0.7, 0.2] and long["correct"].tolist() == [1, 0, 1, 0]
    cm = subject_cell_means(long)
    assert len(cm) == 2 and cm["n_repeats"].tolist() == [2, 2]
