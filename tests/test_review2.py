"""Second-review fixes: Nadeau-Bengio-adjusted omnibus F and the per-representation graph equivalence family."""
import numpy as np
import pandas as pd
import pytest

from eegrep.stats_grid import RATIO_TRAIN, nadeau_bengio
from eegrep.stats_omnibus import nb_adjusted_anova


def _grid(rng, pipelines, edges, effect=None, n=25):
    effect = effect or {}
    rows = []
    for u in range(n):
        base = rng.normal(0.5, 0.05)
        for p in pipelines:
            for e in edges:
                rows.append({"unit": f"{u // 5}-{u % 5}", "pipeline": p, "edge": e,
                             "macro_f1": base + effect.get(p, 0.0) + rng.normal(0, 0.04)})
    return pd.DataFrame(rows)


def test_two_levels_reduce_to_the_nadeau_bengio_t_test(rng):
    df = _grid(rng, ["P1", "P3"], ["spatial", "hybrid"], {"P3": 0.03})
    aov = nb_adjusted_anova(df).set_index("Source")
    marg = df.groupby(["unit", "pipeline"])["macro_f1"].mean().unstack()
    t, p = nadeau_bengio((marg["P3"] - marg["P1"]).to_numpy(), RATIO_TRAIN)
    assert aov.loc["pipeline", "F_nb"] == pytest.approx(t ** 2, rel=1e-9)
    assert aov.loc["pipeline", "p_nb"] == pytest.approx(p, rel=1e-9)
    assert aov.loc["pipeline", "nb_inflation"] == pytest.approx(1 + 25 * RATIO_TRAIN)


def test_adjustment_is_conservative_and_keeps_large_effects(rng):
    df = _grid(rng, [f"P{i}" for i in range(1, 8)], ["spatial", "functional", "hybrid"],
               {"P3": 0.25, "P4": -0.25})
    aov = nb_adjusted_anova(df).set_index("Source")
    assert (aov["p_nb"] >= aov["p-unc"] - 1e-15).all()
    assert aov.loc["pipeline", "p_nb_GG"] < 0.05            # a very large effect survives the correction
    null = nb_adjusted_anova(_grid(rng, [f"P{i}" for i in range(1, 8)], ["spatial", "hybrid"])).set_index("Source")
    assert null.loc["pipeline", "p_nb"] > 0.05


def test_equivalence_has_per_representation_graph_family(tmp_path, rng):
    import json

    from eegrep.stats_equivalence import build

    rep = tmp_path / "reports"
    (rep / "grid").mkdir(parents=True); (rep / "confirm").mkdir()
    df = _grid(rng, ["P1", "P3"], ["spatial", "functional", "hybrid"])
    with open(rep / "grid" / "runs_s0.jsonl", "w") as f:
        for i, r in df.iterrows():
            seed, fold = map(int, r.unit.split("-"))
            f.write(json.dumps({"tag": "main", "pipeline": r.pipeline, "edge": r.edge, "seed": seed, "fold": fold,
                                "config_hash": "x", "subject": {"macro_f1": r.macro_f1, "confusion": []},
                                "window": {"macro_f1": r.macro_f1}, "seconds": 1.0, "epochs_run": 1,
                                "preds": {}}) + "\n")
    pd.DataFrame({"pipeline": ["P1", "P3"] * 25, "model": "rf", "seed": [i // 10 for i in range(50)],
                  "fold": [(i // 2) % 5 for i in range(50)], "macro_f1": rng.normal(0.5, 0.05, 50)}
                 ).to_csv(rep / "confirm" / "classical_train.csv", index=False)
    tab = build(rep)
    fam = tab[tab.family == "graph construction (per representation)"]
    assert len(fam) == 2 * 3 and set(fam.setting) == {"selection"}
    assert (fam.p_tost_holm >= fam.p_tost - 1e-12).all()
