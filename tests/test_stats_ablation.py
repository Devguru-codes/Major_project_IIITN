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


def _factorial(rng, effects):
    return pd.concat([_runs(rng, "main", p, e, m + d) for p, m in effects for e, d in
                      (("spatial", 0.0), ("hybrid", -0.005), ("functional", 0.002))])


def test_rm_anova_detects_representation_effect(rng):
    aov = rm_anova(_factorial(rng, (("P1", 0.55), ("P3", 0.58), ("P4", 0.43)))).set_index("Source")
    assert aov.loc["pipeline", "p-GG-corr"] < 1e-6 and aov.loc["pipeline", "np2"] > 0.9
    assert aov.loc["edge", "p-unc"] > 1e-4 and 0 < aov.loc["edge", "eps-GG"] <= 1


def test_rm_anova_matches_statsmodels(rng):
    from statsmodels.stats.anova import AnovaRM

    df = _factorial(rng, (("P1", 0.55), ("P3", 0.56), ("P4", 0.54)))
    ours = rm_anova(df).set_index("Source")
    ref = AnovaRM(df, "macro_f1", "unit", within=["pipeline", "edge"]).fit().anova_table
    for src, ref_src in (("pipeline", "pipeline"), ("edge", "edge"), ("pipeline * edge", "pipeline:edge")):
        assert ours.loc[src, "F"] == pytest.approx(ref.loc[ref_src, "F Value"], rel=1e-6)
        assert ours.loc[src, "p-unc"] == pytest.approx(ref.loc[ref_src, "Pr > F"], rel=1e-5)


def test_permutation_uses_first_repeat_of_confirmation_folds(rng):
    confirm = _runs(rng, "main", "P3", "spatial", 0.58, seeds=range(5, 10))
    null = pd.concat([_runs(rng, "perm@P3xspatial", "P3", "spatial", 0.33, seeds=[11_000 + i]) for i in range(20)])
    r = permutation_test(confirm, null)["P3xspatial"]
    expected = confirm[confirm.seed == 5]["macro_f1"].mean()
    assert r["observed_seed0_macro_f1"] == pytest.approx(expected)


def test_fair_comparisons_have_both_training_regimes(rng):
    from eegrep.stats_ablation import fair_comparisons

    main = pd.concat([_runs(rng, "main", "P1", e, 0.55) for e in ("spatial", "hybrid")])
    refit = pd.concat([_runs(rng, "refit", "P1", e, 0.57) for e in ("spatial", "hybrid")])
    cls = pd.concat([_runs(rng, "x", "P1", "-", 0.56).assign(model=m) for m in ("lr", "rf")])
    out = fair_comparisons(main, refit, cls, cls)
    assert set(out["training_subjects"]) == {"train only", "train + val"} and len(out) == 2


def test_es_sensitivity_pairs_variants_with_main(rng):
    from eegrep.stats_sensitivity import MAIN_RULE, es_sensitivity

    effects = (("P1", 0.55), ("P3", 0.58), ("P4", 0.43))
    main = _factorial(rng, effects).assign(epochs_run=30, best_epoch=10)
    fixed = _factorial(rng, tuple((p, m + 0.05) for p, m in effects)).assign(epochs_run=100, best_epoch=100)
    cls = pd.concat([_runs(rng, "x", p, "-", 0.5) for p, _ in effects]).assign(model="rf")
    res = es_sensitivity(main, {"fixed 100 epochs": fixed}, cls)
    s = res["summary"].set_index("rule")
    assert list(s.index) == [MAIN_RULE, "fixed 100 epochs"]
    assert s.loc["fixed 100 epochs", "median_best_epoch"] == 100 and s.loc[MAIN_RULE, "best_cell"].startswith("P3")
    assert s.loc["fixed 100 epochs", "spearman_vs_main"] > 0.8 and s.loc["fixed 100 epochs", "rep_p"] < 1e-6
    r = res["representations"].set_index(["rule", "pipeline"])
    assert r.loc[("fixed 100 epochs", "P3"), "delta_vs_main"] == pytest.approx(0.05, abs=0.02)
    assert r.loc[(MAIN_RULE, "P3"), "delta_vs_main"] == 0 and len(res["gcn_vs_classical"]) == 6
