import numpy as np
import pandas as pd
import pytest

from eegrep.ablations import ABLATIONS, run_ablation, run_permutations
from eegrep.config import apply_overrides
from eegrep.runner import ResultStore
from eegrep.splits import make_folds
from eegrep.train import residualize


@pytest.fixture
def setup(cfg, small_cache, tmp_path):
    subjects = sorted(set(small_cache.subject))
    labels = [int(small_cache.label[small_cache.subject == s][0]) for s in subjects]
    folds = make_folds(subjects, labels, [0], n_folds=3, val_frac=0.34)
    part = pd.DataFrame({"subject": subjects, "age": np.linspace(55, 80, len(subjects)),
                         "sex_male": [i % 2 for i in range(len(subjects))]})
    return apply_overrides(cfg, {"train.max_epochs": 1}), folds, part, ResultStore(tmp_path / "abl.jsonl")


@pytest.mark.parametrize("name", list(ABLATIONS))
def test_every_ablation_runs(setup, small_cache, name):
    cfg, folds, part, store = setup
    n = run_ablation(name, small_cache, folds, cfg, store, pipeline="P1", edge="hybrid", participants=part)
    assert n == 3


def test_permutation_runs_unique_keys(setup, small_cache):
    cfg, folds, _, store = setup
    assert run_permutations(small_cache, folds, cfg, store, pipeline="P1", edge="hybrid", n_perm=2) == 6
    assert len(store.done) == 6


def test_select_cells_ranks_main_runs_only(tmp_path):
    import json

    from eegrep.ablations import select_cells

    rows = [("main", "P1", "hybrid", 0.6), ("main", "P1", "hybrid", 0.7), ("main", "P6", "spatial", 0.8),
            ("main", "P2", "functional", 0.5), ("bench", "P5", "hybrid", 0.99)]
    f = tmp_path / "runs.jsonl"
    f.write_text("\n".join(json.dumps({"tag": t, "pipeline": p, "edge": e, "subject": {"macro_f1": v}})
                           for t, p, e, v in rows))
    assert select_cells([f], top=2) == ["P6xspatial", "P1xhybrid"]


def test_residualize_removes_linear_covariate_effect(rng):
    cov = rng.normal(size=(200, 1))
    X = (3.0 * cov[:, :, None] + rng.normal(scale=0.1, size=(200, 2, 2))).astype(np.float32)
    r = residualize(X, cov, np.arange(150))
    assert abs(np.corrcoef(cov[:, 0], r[:, 0, 0])[0, 1]) < 0.2
