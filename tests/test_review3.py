"""Third-review checks: calibration simulation of the NB-adjusted omnibus F; occlusion re-fits reproduce stored runs."""
import json

import numpy as np
import pandas as pd
import pytest

from eegrep.stats_grid import RATIO_TRAIN, nadeau_bengio, rm_anova
from eegrep.stats_simulation import oneway_rm, simulate, summarise


def test_oneway_matches_two_way_main_effect_and_nb_t(rng):
    rows = []
    for u in range(25):
        b = rng.normal(0.5, 0.05)
        for p, eff in (("P1", 0.0), ("P2", 0.02), ("P3", 0.05)):
            for e in ("spatial", "hybrid"):
                rows.append({"unit": str(u), "pipeline": p, "edge": e, "macro_f1": b + eff + rng.normal(0, 0.03)})
    df = pd.DataFrame(rows)
    marg = df.groupby(["unit", "pipeline"])["macro_f1"].mean().unstack().to_numpy()
    two = rm_anova(df).set_index("Source")
    assert oneway_rm(marg)["F"] == pytest.approx(two.loc["pipeline", "F"], rel=1e-9)
    d = marg[:, 2] - marg[:, 0]
    t, _ = nadeau_bengio(d, RATIO_TRAIN)
    assert oneway_rm(marg[:, [0, 2]])["F_nb"] == pytest.approx(t ** 2, rel=1e-9)


def test_simulation_runs_and_reports_rejection_rates():
    sims = simulate(n_sim=3, k=3, spread=0.0, seed=1)
    assert len(sims) == 3 and sims["mean_f1"].between(0, 1).all()
    s = summarise(sims)
    assert {"reject_uncorrected", "reject_nb"} <= set(s.columns)
    assert (sims.p_nb_gg >= sims.p_gg - 1e-12).all()


def test_refit_reproduces_stored_run(cfg, small_cache, tmp_path):
    from eegrep.explain import refit_check
    from eegrep.runner import ResultStore, run_cell
    from eegrep.splits import make_folds
    from eegrep.stats_grid import load_runs

    subjects = sorted(set(small_cache.subject))
    labels = [int(small_cache.label[small_cache.subject == s][0]) for s in subjects]
    folds = make_folds(subjects, labels, [0], n_folds=3, val_frac=0.34)
    quick = {"train.max_epochs": 3}
    run_cell(small_cache, folds, cfg, ResultStore(tmp_path / "r.jsonl"), pipeline="P1", edge="spatial",
             tag="main", overrides=quick)
    from eegrep.config import apply_overrides

    stored = load_runs([tmp_path / "r.jsonl"], tag="main")
    tab = refit_check(small_cache, folds, apply_overrides(cfg, quick), stored, "P1", "spatial")
    assert len(tab) == 3
    assert (tab.max_abs_prob_diff < 1e-4).all() and (tab.argmax_agreement == 1.0).all()
