"""Subject-level mixed model: recovers a known representation effect, stays calm under the null, and reshapes runs."""
import numpy as np
import pandas as pd
import pytest

from eegrep.stats_mixed import analyse_mixed, subject_long


def _simulate(rng, rep_effect, n_subj=30, reps=("P1", "P2", "P3"), edges=("hybrid", "spatial"), repeats=2, folds=3):
    subj = [f"sub-{i:03d}" for i in range(n_subj)]
    cls = {s: i % 3 for i, s in enumerate(subj)}
    u_subj = {s: rng.normal(0, 0.10) for s in subj}
    rows = []
    for r in range(repeats):
        order = rng.permutation(subj)
        for k in range(folds):
            unit = f"{r}-{k}"
            u_split = rng.normal(0, 0.03)
            for p in reps:
                for e in edges:
                    u_model = rng.normal(0, 0.03)
                    for s in order[k::folds]:
                        y = 0.45 + rep_effect[p] + u_subj[s] + u_split + u_model + rng.normal(0, 0.08)
                        rows.append((s, cls[s], y, int(y > 0.5), p, e, unit, f"{p}|{e}|{unit}"))
    return pd.DataFrame(rows, columns=["subject", "true_class", "p_true", "correct", "pipeline", "edge", "unit", "model"])


def test_recovers_representation_effect(rng):
    res = analyse_mixed(_simulate(rng, {"P1": 0.0, "P2": -0.08, "P3": 0.06}))
    t = res["terms"].set_index("term")
    assert t.loc["representation", "p"] < 1e-6 and t.loc["representation", "df"] == 2
    assert t.loc["graph", "df"] == 1 and t.loc["interaction", "df"] == 2
    m = res["marginal"].query("factor == 'pipeline'").set_index("level")["mean"]
    assert m["P3"] > m["P1"] > m["P2"]
    assert m["P3"] - m["P2"] == pytest.approx(0.14, abs=0.03)
    pw = res["pairwise_representation"].set_index(["a", "b"])
    assert pw.loc[("P2", "P3"), "p_holm"] < 0.01
    vc = res["variance_components"]
    assert {"subject", "split", "model", "residual"} <= set(vc) and vc["subject"]["share"] > vc["split"]["share"]


def test_null_false_positive_rate_is_nominal():
    """Calibration under the null: across 30 simulated datasets the representation test rejects at about 5%."""
    p = [analyse_mixed(_simulate(np.random.default_rng(100 + i), {"P1": 0.0, "P2": 0.0, "P3": 0.0}))["terms"]
         .set_index("term").loc["representation", "p"] for i in range(30)]
    assert np.mean(np.array(p) < 0.05) <= 0.15
    assert np.mean(p) > 0.3                                            # roughly uniform, not piled near zero


def test_subject_long_reshapes_predictions():
    runs = pd.DataFrame([{"pipeline": "P1", "edge": "spatial", "seed": 0, "fold": 1,
                          "preds": {"subjects": ["a", "b"], "y": [0, 2], "prob": [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]]}}])
    long = subject_long(runs)
    assert long["p_true"].tolist() == [0.7, 0.2] and long["correct"].tolist() == [1, 0]
    assert set(long["unit"]) == {"0-1"} and set(long["model"]) == {"P1|spatial|0-1"}
