"""Equivalence tests (TOST, NB-corrected) and the PSWE grid / collinearity checks."""
import numpy as np
import pandas as pd
import pytest

from eegrep.stats_equivalence import paired


def _units(n=25):
    return [f"{s}-{k}" for s in range(5) for k in range(5)][:n]


def test_tost_equivalent_for_tiny_differences(rng):
    u = _units()
    b = pd.Series(rng.normal(0.55, 0.1, 25), index=u)
    a = b + rng.normal(0.0, 0.01, 25)
    r = paired(a, b, 17.6 / 56.3)
    assert r["p_tost"] < 0.05 and -0.05 < r["ci90_lo"] < r["ci90_hi"] < 0.05


def test_tost_not_equivalent_for_large_or_noisy_differences(rng):
    u = _units()
    b = pd.Series(rng.normal(0.55, 0.1, 25), index=u)
    assert paired(b + 0.10, b, 17.6 / 56.3)["p_tost"] > 0.05                       # real difference
    assert paired(b + rng.normal(0, 0.08, 25), b, 17.6 / 56.3)["p_tost"] > 0.05   # too noisy to tell


def test_mde_matches_formula(rng):
    from scipy import stats

    u = _units()
    b = pd.Series(rng.normal(0.55, 0.1, 25), index=u)
    d = rng.normal(0, 0.05, 25)
    r = paired(b + d, b, 0.3)
    se = np.sqrt((1 / 25 + 0.3) * d.var(ddof=1))
    assert r["se_nb"] == pytest.approx(se) and r["mde80"] == pytest.approx((stats.t.ppf(0.975, 24) + stats.t.ppf(0.8, 24)) * se)


def _subjects(rng, slowing_by_group):
    rows = []
    for g, (n, mu) in enumerate(slowing_by_group):
        for i in range(n):
            slow = rng.normal(mu, 0.05)
            rows.append({"subject": f"s{g}-{i}", "label": g, "age": rng.normal(65, 5), "sex_male": i % 2,
                         "rel_delta_theta": slow, "duration_s": 600.0,
                         "n_events": rng.poisson(np.exp(1.0 + 6 * slow))})
    return pd.DataFrame(rows).set_index("subject")


def test_collinearity_vif_and_observed_expected(rng):
    from eegrep.stats_pswe import collinearity

    names = ["AD", "FTD", "CN"]
    indep = collinearity(_subjects(rng, [(30, 0.4), (30, 0.4), (30, 0.4)]), names)
    coll = collinearity(_subjects(rng, [(30, 0.7), (30, 0.5), (30, 0.3)]), names)
    assert indep["vif_slowing"] < 1.5 < coll["vif_slowing"]
    assert indep["observed_expected"]["CN"] == pytest.approx(1.0, abs=0.25)
    assert coll["common_support"]["n_in_support"]["AD"] < coll["common_support"]["n_total"]["AD"]


def test_grid_cell_at_main_rule_reproduces_detector(cfg, tmp_path):
    from eegrep import synthetic
    from eegrep.pswe import detect
    from eegrep.stats_pswe import grid_tables

    recs, _ = synthetic.make_recordings(n_per_class=1, duration_s=60.0, seed=3)
    counts = {}
    for subject, data, sfreq in recs:
        np.savez(tmp_path / f"{subject}.npz", data=data, sfreq=sfreq)
        ev, *_ = detect(data, sfreq, cfg["pswe"], cfg["dataset"]["channels"])
        counts[subject] = len(ev)
    tabs = grid_tables([tmp_path], sorted(counts), cfg)
    main = tabs[(cfg["pswe"]["mpf_threshold_hz"], cfg["pswe"]["min_duration_s"])]
    assert main["n_events"].to_dict() == counts and len(tabs) == 9
    # a looser rule covers at least the time of a stricter one (counts can fall when events merge)
    assert all(tabs[(7.0, 3.0)]["time_fraction"] >= tabs[(5.0, 8.0)]["time_fraction"])
