import numpy as np
import pandas as pd
import pytest

from eegrep.stats_grid import critical_difference, nadeau_bengio, pairwise
from eegrep.stats_pswe import dunn, holm


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_nadeau_bengio_is_more_conservative_than_naive_t(rng):
    d = rng.normal(0.02, 0.05, 25)
    t_nb, p_nb = nadeau_bengio(d, 0.25)
    t_naive = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(t_nb) < abs(t_naive) and 0 <= p_nb <= 1


def test_pairwise_detects_clear_difference(rng):
    base = rng.normal(0.6, 0.05, 25)
    wide = pd.DataFrame({"good": base + 0.15, "bad": base, "same": base + rng.normal(0, 0.01, 25)})
    pw = pairwise(wide, 0.25).set_index(["a", "b"])
    assert pw.loc[("good", "bad"), "wilcoxon_p_holm"] < 0.001
    assert pw.loc[("good", "bad"), "rank_biserial"] == pytest.approx(1.0)


def test_critical_difference_ranks_best_first(rng):
    base = rng.normal(0.6, 0.05, (25, 1))
    wide = pd.DataFrame(base + np.array([[0.2, 0.1, 0.0]]) + rng.normal(0, 0.01, (25, 3)), columns=["a", "b", "c"])
    cd = critical_difference(wide)
    assert list(cd["avg_rank"]) == ["a", "b", "c"] and cd["friedman_p"] < 0.001 and cd["cd"] > 0


def test_dunn_separates_shifted_group(rng):
    v = np.concatenate([rng.normal(0, 1, 30), rng.normal(3, 1, 30), rng.normal(0, 1, 30)])
    g = np.repeat([0, 1, 2], 30)
    res = {r["pair"]: r for r in dunn(v, g, {0: "AD", 1: "FTD", 2: "CN"})}
    assert res["AD vs FTD"]["p_holm"] < 0.001 and res["AD vs CN"]["p_holm"] > 0.05
