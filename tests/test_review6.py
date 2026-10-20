"""Sixth-review analyses: data-matched null simulation and the out-of-bootstrap representation comparison."""
import json

import numpy as np
import pandas as pd
import pytest

from eegrep.stats_simulation import calibrate, calibrated_null, observed_targets, split_stats


def test_split_stats_and_observed_targets(rng):
    rows = [{"unit": f"{u // 5}-{u % 5}", "pipeline": p, "edge": e,
             "macro_f1": 0.5 + 0.02 * i + rng.normal(0, 0.05)}
            for u in range(25) for i, p in enumerate(["P1", "P2", "P3"]) for e in ("spatial", "hybrid")]
    t = observed_targets(pd.DataFrame(rows))
    assert set(t) == {"mean_f1", "split_sd", "rep_corr", "range_f1"}
    assert 0.45 < t["mean_f1"] < 0.6 and t["range_f1"] == pytest.approx(0.04, abs=0.03)
    one = observed_targets(pd.DataFrame(rows), edge="spatial")            # one graph: fewer runs averaged
    assert one["split_sd"] > t["split_sd"]
    s = split_stats(np.column_stack([np.arange(25.0), np.arange(25.0) * 2]))
    assert s["rep_corr"] == pytest.approx(1.0)


def test_calibration_picks_closest_setting_and_null_p_value():
    targets = {"mean_f1": 0.5, "split_sd": 0.1, "rep_corr": 0.5, "range_f1": 0.13}
    grid = calibrate(targets, separations=(0.5, 1.0), noises=(1.0,), corrs=(0.5,), n_sim=1, k=3)
    assert len(grid) == 2 and grid.distance.is_monotonic_increasing
    sims, summ = calibrated_null(targets, 0.5, 1.0, 0.5, n_sim=3, k=3)
    assert len(sims) == 3 and 0 < summ["p_range"] <= 1 and summ["observed_range"] == 0.13
    assert summ["observed_std_range"] == pytest.approx(1.3) and 0 < summ["p_std_range"] <= 1
    assert summ["null_std_range_mean"] == pytest.approx((sims.range_f1 / sims.split_sd).mean())


def test_bootstrap_split_is_stratified_and_disjoint():
    from eegrep.stats_bootstrap import bootstrap_split

    subjects = np.array([f"s{i:02d}" for i in range(30)]); labels = np.repeat([0, 1, 2], 10)
    tr, va, oob = bootstrap_split(subjects, labels, b=3)
    assert not (set(tr) & set(va)) and not ((set(tr) | set(va)) & set(oob))
    assert set(tr) | set(va) | set(oob) == set(subjects) and len(oob) > 0


def test_bootstrap_runs_and_summarises(cfg, small_cache, tmp_path):
    from eegrep.config import apply_overrides
    from eegrep.stats_bootstrap import run_bootstrap, summarise

    quick = apply_overrides(cfg, {"train.max_epochs": 2})
    out = tmp_path / "boot.jsonl"
    assert run_bootstrap(small_cache, quick, ["P1", "P3"], "spatial", range(3), out) == 6
    assert run_bootstrap(small_cache, quick, ["P1", "P3"], "spatial", range(3), out) == 0     # resumes
    rows = pd.DataFrame([json.loads(line) for line in open(out)])
    reps, pairs = summarise(rows)
    assert set(reps.pipeline) == {"P1", "P3"} and len(pairs) == 1
    assert (pairs.p_holm >= pairs.p_boot - 1e-12).all() and pairs.lo.iloc[0] <= pairs.hi.iloc[0]
