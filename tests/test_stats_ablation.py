import numpy as np
import pandas as pd
import pytest

from eegrep.stats_ablation import ablation_table, leakage_table, permutation_test
from eegrep.stats_grid import rm_anova


def _runs(rng, tag, p, e, mean, seeds=range(5), folds=range(5), window=None):
    rows = []
    for s in seeds:
        for k in folds:
            f1 = mean + rng.normal(0, 0.02)
            rows.append({"tag": tag, "pipeline": p, "edge": e, "seed": s, "fold": k, "unit": f"{s}-{k}",
                         "macro_f1": f1, "window_macro_f1": window if window is not None else f1 - 0.05,
                         "accuracy": f1})
    return pd.DataFrame(rows)


def test_ablation_deltas_are_paired_and_holm_corrected(rng):
    main = _runs(rng, "main", "P3", "spatial", 0.60)
    abl = pd.concat([_runs(rng, "A1_no_graph@P3xspatial", "P3", "identity", 0.50),
                     _runs(rng, "A3_k6@P3xspatial", "P3", "spatial", 0.60)])
    t = ablation_table(main, abl).set_index("ablation")
    assert t.loc["A1_no_graph", "delta"] == pytest.approx(-0.10, abs=0.02)
    assert t.loc["A1_no_graph", "nb_p_holm"] < 0.05 and t.loc["A3_k6", "nb_p_holm"] > 0.05
    assert t.loc["A1_no_graph", "delta_ci_hi"] < 0


def test_leakage_table_reports_inflation(rng):
    main = _runs(rng, "main", "P3", "spatial", 0.58)
    leak = _runs(rng, "A10_window_split@P3xspatial", "P3", "spatial", 0.98, window=0.84)
    t = leakage_table(main, leak)
    assert set(t["protocol"]) == {"subject-level (correct)", "window-level (leaky)"}
    assert t["inflation_window_vs_correct"].iloc[0] == pytest.approx(0.84 - 0.58, abs=0.02)


def test_permutation_p_value(rng):
    main = _runs(rng, "main", "P3", "spatial", 0.58)
    null = pd.concat([_runs(rng, "perm@P3xspatial", "P3", "spatial", 0.33, seeds=[10_000 + i]) for i in range(50)])
    r = permutation_test(main, null)["P3xspatial"]
    assert r["n_perm"] == 50 and r["p"] == pytest.approx(1 / 51)


def test_rm_anova_detects_representation_effect(rng):
    df = pd.concat([_runs(rng, "main", p, e, m + d)
                    for p, m in (("P1", 0.55), ("P3", 0.58), ("P4", 0.43))
                    for e, d in (("spatial", 0.0), ("hybrid", -0.005))])
    aov = rm_anova(df)
    p_col = "p-unc" if "p-unc" in aov.columns else aov.columns[-1]
    src = aov.set_index("Source")
    assert src.loc["pipeline", p_col] < 1e-6
