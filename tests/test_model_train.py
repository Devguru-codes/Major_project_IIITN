import numpy as np
import pytest
import torch

from eegrep.config import apply_overrides
from eegrep.models import build_model, n_params
from eegrep.runner import ResultStore, run_cell
from eegrep.splits import make_folds
from eegrep.train import fit_scaler, window_weights


@pytest.mark.parametrize("f", [7, 20])
def test_gcn_is_about_3k_params(cfg, f):
    assert 2500 <= n_params(build_model(f, cfg["model"])) <= 3700


@pytest.mark.parametrize("arch", ["gcn", "gat"])
@pytest.mark.parametrize("hidden", [[32], [64, 32], [64, 64, 32]])
def test_forward_shared_and_batched_adjacency(cfg, arch, hidden):
    m = build_model(7, {**cfg["model"], "arch": arch, "hidden": hidden}).eval()
    x = torch.randn(8, 19, 7)
    a = torch.eye(19) * 0.5 + 0.5 / 19
    assert m(x, a).shape == (8, 3)
    assert m(x, a.expand(8, 19, 19)).shape == (8, 3)


def test_scaler_uses_training_windows_only(rng):
    x = rng.standard_normal((10, 19, 4)).astype(np.float32)
    x[5:] += 100.0                                              # test windows: very different
    mu, _ = fit_scaler(x[:5])
    assert np.allclose(mu, x[:5].reshape(-1, 4).mean(0))
    assert np.all(mu < 10)


def test_subject_equal_weighting():
    subj = np.array(["a"] * 2 + ["b"] * 8 + ["c"] * 5)
    lab = np.array([0] * 2 + [0] * 8 + [1] * 5)
    w = window_weights(subj, lab, 2, subject_equal=True)
    assert w[subj == "a"].sum() == pytest.approx(w[subj == "b"].sum())


def test_runner_trains_and_resumes(cfg, small_cache, tmp_path):
    subjects = sorted(set(small_cache.subject))
    labels = [int(small_cache.label[small_cache.subject == s][0]) for s in subjects]
    folds = make_folds(subjects, labels, [0], n_folds=3, val_frac=0.34)
    store = ResultStore(tmp_path / "runs.jsonl")
    quick = {"train.max_epochs": 2}
    n = run_cell(small_cache, folds, cfg, store, pipeline="P1", edge="hybrid", overrides=quick)
    assert n == 3
    again = run_cell(small_cache, folds, cfg, ResultStore(tmp_path / "runs.jsonl"),
                     pipeline="P1", edge="hybrid", overrides=quick)
    assert again == 0
    changed = run_cell(small_cache, folds, apply_overrides(cfg, {"graph.knn_k": 3}), store,
                       pipeline="P1", edge="hybrid", overrides=quick)
    assert changed == 3                                          # config change => new keys
