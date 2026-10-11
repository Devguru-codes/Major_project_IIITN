"""Should-fix additions: learned-adjacency GCN, occlusion importance, by-subject RM-ANOVA cross-check."""
import json

import numpy as np
import pytest
import torch

from eegrep.config import apply_overrides
from eegrep.models import LearnedAdjGCN, build_model
from eegrep.runner import ResultStore, run_cell
from eegrep.splits import make_folds


def _folds(cache):
    subjects = sorted(set(cache.subject))
    labels = [int(cache.label[cache.subject == s][0]) for s in subjects]
    return make_folds(subjects, labels, [0], n_folds=3, val_frac=0.34)


def test_learned_adjacency_initialises_from_graph_and_learns(cfg):
    m = build_model(7, {**cfg["model"], "arch": "gcn_learned"}, 3, n_nodes=19)
    assert isinstance(m, LearnedAdjGCN)
    a = torch.rand(19, 19)
    a = (a + a.T) / 2
    out = m(torch.randn(8, 19, 7), a)
    w = m.learned_adjacency()
    assert out.shape == (8, 3) and torch.allclose(w, w.T, atol=1e-6) and (w >= 0).all()
    assert bool(m.initialised)
    out.sum().backward()
    assert m.adj_logit.grad is not None and m.adj_logit.grad.abs().sum() > 0


def test_learned_adjacency_cell_runs(cfg, small_cache, tmp_path):
    store = ResultStore(tmp_path / "learned.jsonl")
    run_cell(small_cache, _folds(small_cache), cfg, store, pipeline="P1", edge="spatial", tag="learned_adj",
             overrides={"model.arch": "gcn_learned", "train.max_epochs": 2})
    rows = [json.loads(line) for line in open(tmp_path / "learned.jsonl")]
    assert len(rows) == 3 and all(r["n_params"] > 2_819 for r in rows)          # + 19x19 adjacency


def test_occlusion_covers_every_electrode_and_band(cfg, small_cache):
    from eegrep.explain import occlusion, summarise

    quick = apply_overrides(cfg, {"train.max_epochs": 2})
    occ = occlusion(small_cache, _folds(small_cache), quick, "P3", "spatial")
    n_test = len(set(small_cache.subject))                                       # each subject tested once
    assert len(occ) == n_test * (19 + 5)
    s = summarise(occ, cfg["dataset"]["class_names"])
    assert set(s.kind) == {"electrode", "band"} and {"drop_AD", "drop_FTD", "drop_CN"} <= set(s.columns)
    assert (s.ci_lo <= s["drop"] + 1e-12).all() and (s["drop"] <= s.ci_hi + 1e-12).all()


def test_mixed_reports_by_subject_rm_anova(rng):
    from test_mixed import _simulate        # pytest puts tests/ on sys.path (no package __init__)

    from eegrep.stats_mixed import analyse_mixed

    res = analyse_mixed(_simulate(rng, {"P1": 0.0, "P2": -0.05, "P3": 0.03}), with_class=False)
    aov = res["rm_anova"].set_index("Source")
    t = res["terms"].set_index("term")
    assert aov.loc["pipeline", "F"] == pytest.approx(t.loc["representation", "F"], rel=0.02)
    assert 0 < aov.loc["pipeline", "eps-GG"] <= 1
