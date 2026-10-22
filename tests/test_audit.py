"""Repo audit fixes (2026-10-22): random-graph ablation base, download pinning, resume, equivalence edge case."""
import json

import numpy as np
import pandas as pd
import pytest

from eegrep import graphs
from eegrep.data.download import check_download
from eegrep.runner import ResultStore
from eegrep.stats_equivalence import paired


def _degrees(a):
    return sorted(((a - np.diag(np.diag(a))) > 0).sum(-1).tolist())


def test_random_graph_shuffles_the_reference_graph(cfg, rng):
    coords = graphs.electrode_positions(cfg["dataset"]["channels"], cfg["dataset"]["channel_rename"])
    spatial = graphs.spatial_adjacency(coords)
    func = rng.random((2, 19, 19)).astype(np.float32)
    func = (func + func.transpose(0, 2, 1)) / 2
    r = graphs.build("random_spatial", spatial=spatial, k=4, seed=3)           # no functional graph needed
    assert r.ndim == 2 and _degrees(r) == _degrees(graphs.build("spatial", spatial=spatial, k=4))
    assert not np.allclose(r, graphs.build("spatial", spatial=spatial, k=4))
    old = graphs.build("random", spatial=spatial, functional=func, k=4, seed=3)  # stored hybrid-cell runs
    assert np.allclose(old, graphs.build("random_hybrid", spatial=spatial, functional=func, k=4, seed=3))


def test_a2_uses_the_reference_cells_graph(cfg, small_cache, tmp_path):
    from eegrep.ablations import run_ablation
    from eegrep.config import apply_overrides
    from eegrep.splits import make_folds

    subjects = sorted(set(small_cache.subject))
    labels = [int(small_cache.label[small_cache.subject == s][0]) for s in subjects]
    folds = make_folds(subjects, labels, [0], n_folds=3, val_frac=0.34)
    store = ResultStore(tmp_path / "a2.jsonl")
    c = apply_overrides(cfg, {"train.max_epochs": 1})
    for edge in ("spatial", "hybrid"):
        run_ablation("A2_random_graph", small_cache, folds, c, store, pipeline="P1", edge=edge)
    edges = {json.loads(l)["tag"]: json.loads(l)["edge"] for l in open(tmp_path / "a2.jsonl")}
    assert edges == {"A2_random_graph@P1xspatial": "random_spatial", "A2_random_graph@P1xhybrid": "random"}


def test_download_refuses_other_version_or_changed_files(tmp_path):
    (tmp_path / "CHANGES").write_text("1.0.9 2026-05-20\n  - note\n")
    files = [{"path": "a.set", "size": 1, "sha256": "x"}, {"path": "b.set", "size": 1, "sha256": "y"}]
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"files": [{"path": "a.set", "sha256": "x"}]}))
    check_download(tmp_path, files, "1.0.9", ref)                                 # matches; b.set not in ref
    with pytest.raises(RuntimeError, match="version"):
        check_download(tmp_path, files, "1.0.8", None)
    ref.write_text(json.dumps({"files": [{"path": "a.set", "sha256": "changed"}]}))
    with pytest.raises(RuntimeError, match="differ"):
        check_download(tmp_path, files, None, ref)


def test_resume_drops_only_a_partial_last_line(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text('{"key": "a"}\n{"key": "b"}\n{"key": "c", "sub')
    assert ResultStore(p).done == {"a", "b"}
    assert p.read_text() == '{"key": "a"}\n{"key": "b"}\n'
    p.write_text('{"key": "a"}\n{"ke\n{"key": "c"}\n')                           # damage elsewhere still raises
    with pytest.raises(json.JSONDecodeError):
        ResultStore(p)


def test_constant_difference_is_equivalent_only_inside_margin():
    idx = [f"0-{i}" for i in range(25)]
    a = pd.Series(0.6, index=idx)
    assert paired(a, pd.Series(0.5, index=idx), 0.31)["p_tost"] == 1.0          # every split 0.10 apart
    assert paired(a, pd.Series(0.58, index=idx), 0.31)["p_tost"] == 0.0         # 0.02 apart: inside +/-0.05


def test_qc_flags_cached_subjects_without_a_log(cfg, small_cache, tmp_path):
    from eegrep.qc import gate_failures, qc_report

    subjects = sorted(set(small_cache.subject))
    log = tmp_path / "preprocess_log.jsonl"
    log.write_text("".join(json.dumps({"subject": s, "excluded": [], "excluded_labels": [], "duration_s": 40.0})
                           + "\n" for s in subjects[1:]))
    rep = qc_report(small_cache.dir, [log], cfg["dataset"]["class_names"])
    assert rep["logs_missing"] == subjects[:1]
    assert any("no preprocessing log" in f for f in gate_failures(rep))


def test_replace_ablations_pairs_with_own_reference_and_redoes_holm():
    from eegrep.stats_ablation import replace_ablations

    units = [f"0-{i}" for i in range(25)]
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"cell": ["P3xspatial"] * 2 + ["P3xhybrid"], "ablation": ["A1", "A2", "A2"],
                          "delta": [-0.03, -0.01, -0.004], "delta_ci_lo": 0.0, "delta_ci_hi": 0.0,
                          "wilcoxon_p": [0.02, 0.9, 0.8], "nb_p": [0.01, 0.9, 0.8],
                          "wilcoxon_p_holm": 0.0, "nb_p_holm": 0.0})
    ref_f1 = 0.55 + rng.normal(0, 0.05, 25)
    ref = pd.DataFrame({"pipeline": "P3", "edge": "spatial", "unit": units, "macro_f1": ref_f1, "tag": "main"})
    abl = ref.assign(tag="A2@P3xspatial", edge="random_spatial", macro_f1=ref_f1 + 0.004)
    t = replace_ablations(table, ref, abl, note="rebuilt cache").set_index(["cell", "ablation"])
    assert t.loc[("P3xspatial", "A2"), "delta"] == pytest.approx(0.004)
    assert t.loc[("P3xspatial", "A2"), "source"] == "rebuilt cache" and t.loc[("P3xhybrid", "A2"), "source"] == ""
    assert t.loc[("P3xspatial", "A1"), "nb_p_holm"] == pytest.approx(0.02)        # Holm over the 2 spatial rows
    assert t.loc[("P3xhybrid", "A2"), "delta"] == -0.004                          # other cell untouched
