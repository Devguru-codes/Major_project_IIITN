"""External cohort (ds004584): config overlay, reference restoration, participants mapping, binary training."""
import json

import numpy as np
import pandas as pd
import pytest

from eegrep.config import REPO_ROOT, config_hash, load_config
from eegrep.data.participants import assert_no_mmse, load_participants, model_covariates
from eegrep.metrics import classification_metrics
from eegrep.runner import ResultStore, run_cell
from eegrep.splits import make_folds

EXT = REPO_ROOT / "configs" / "ds004584.yaml"


@pytest.fixture(scope="module")
def ext_cfg():
    return load_config(EXT)


def test_overlay_changes_only_dataset_and_notch(cfg, ext_cfg):
    assert ext_cfg["dataset"]["class_names"] == ["PD", "CN"] and ext_cfg["preprocess"]["notch"] == 60.0
    for section in ("features", "pswe", "graph", "model", "train", "cv", "bands"):
        assert ext_cfg[section] == cfg[section]
    assert {k: v for k, v in ext_cfg["preprocess"].items() if k != "notch"} == \
           {k: v for k, v in cfg["preprocess"].items() if k != "notch"}
    assert config_hash(ext_cfg) != config_hash(cfg)                      # notch is part of the hash
    # same electrodes, same node order (standard_1020 names)
    assert ext_cfg["dataset"]["channels"] == [cfg["dataset"]["channel_rename"].get(c, c) for c in cfg["dataset"]["channels"]]


def test_overlay_rejects_unknown_keys(tmp_path):
    (tmp_path / "bad.yaml").write_text(f"base: {(REPO_ROOT / 'configs' / 'default.yaml').as_posix()}\n"
                                       "train: {learning_rate: 0.1}\n")
    with pytest.raises(KeyError):
        load_config(tmp_path / "bad.yaml")


def test_reference_channel_restored_flat_and_ordered(ext_cfg):
    import mne

    from eegrep.preprocess import prepare_channels

    stored = [c for c in ext_cfg["dataset"]["channels"] if c != "Pz"] + ["FCz", "AF3"]   # Pz is the reference
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.normal(0, 1e-5, (len(stored), 1000)), mne.create_info(stored, 500.0, "eeg"),
                          verbose="error")
    out = prepare_channels(raw, ext_cfg["dataset"])
    assert out.ch_names == ext_cfg["dataset"]["channels"]
    assert np.allclose(out.get_data(picks=["Pz"]), 0.0)
    assert not np.allclose(out.get_data(picks=["Fz"]), 0.0)


def test_participants_mapping_and_firewall(ext_cfg, tmp_path):
    (tmp_path / "p.tsv").write_text("participant_id\tGROUP\tID\tEEG\tAGE\tGENDER\tMOCA\tUPDRS\tTYPE\n"
                                    "sub-001\tPD\t1\tx\t70\tM\t24\t20\t1\n"
                                    "sub-002\tControl\t2\ty\t68\tF\t28\tn/a\t0\n")
    d = ext_cfg["dataset"]
    df = load_participants(tmp_path / "p.tsv", d["group_to_label"], d["participant_columns"])
    assert df["label"].tolist() == [0, 1] and df["sex_male"].tolist() == [1, 0] and df["mmse"].tolist() == [24, 28]
    assert set(model_covariates(df).columns) == {"subject", "age", "sex_male"}
    for col in ("MOCA", "UPDRS"):
        with pytest.raises(AssertionError):
            assert_no_mmse(["age", col])


def test_binary_metrics_have_auc():
    y = np.array([0, 0, 1, 1, 1])
    prob = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])
    m = classification_metrics(y, prob, 2)
    assert m["roc_auc"] == pytest.approx(5 / 6) and "sens_2" not in m


def test_binary_cell_trains_two_class_model(ext_cfg, small_cache, tmp_path):
    binary = (small_cache.label > 0).astype(int)                           # stand-in two-class labelling
    subjects = sorted(set(small_cache.subject))
    labels = [int(binary[small_cache.subject == s][0]) for s in subjects]
    folds = make_folds(subjects, labels, [0], n_folds=3, val_frac=0.34)
    store = ResultStore(tmp_path / "bin.jsonl")
    run_cell(small_cache, folds, ext_cfg, store, pipeline="P1", edge="spatial", tag="main", labels=binary,
             overrides={"train.max_epochs": 2})
    rows = [json.loads(line) for line in open(tmp_path / "bin.jsonl")]
    assert len(rows) == 3 and all(np.array(r["subject"]["confusion"]).shape == (2, 2) for r in rows)
    assert all(len(r["preds"]["prob"][0]) == 2 for r in rows)


def test_ranking_transfer():
    from eegrep.stats_external import transfer

    idx = [(p, e) for p in ("P1", "P2", "P3") for e in ("spatial", "hybrid")]
    src = pd.DataFrame([{"pipeline": p, "edge": e, "macro_f1_mean": v} for (p, e), v in zip(idx, [.5, .45, .4, .38, .6, .58])])
    ext = src.assign(macro_f1_mean=src["macro_f1_mean"] * 0.8 + 0.2)          # same order
    t = transfer(src, ext)
    assert t["cells"]["spearman_rho"] == pytest.approx(1.0) and t["representations"]["kendall_tau"] == pytest.approx(1.0)


def test_qc_gates_follow_config(ext_cfg):
    from eegrep.qc import gate_failures

    rep = {"n_subjects": 149, "n_windows": 2300, "duration_s": {"min": 120.6, "max": 342.7},
           "finite": {"P1": True}, "theta_alpha_ratio": {"PD": 0.5, "CN": 0.6}}
    assert gate_failures(rep, ext_cfg["dataset"]["qc"]) == []
    assert gate_failures({**rep, "n_subjects": 88}, ext_cfg["dataset"]["qc"])


@pytest.mark.parametrize("name, key, value", [("ablation_a7_window5", "window_s", 5.0),
                                               ("ablation_a7_window20", "window_s", 20.0),
                                               ("ablation_a8_no_ica", "ica_method", "none")])
def test_a7_a8_overlays_change_one_key(cfg, name, key, value):
    abl = load_config(REPO_ROOT / "configs" / f"{name}.yaml")
    assert abl["preprocess"][key] == value
    assert {k: v for k, v in abl["preprocess"].items() if k != key} == \
           {k: v for k, v in cfg["preprocess"].items() if k != key}
    assert {s: abl[s] for s in abl if s != "preprocess"} == {s: cfg[s] for s in cfg if s != "preprocess"}
    assert config_hash(abl) != config_hash(cfg)


def test_no_ica_leaves_data_untouched(cfg):
    import mne

    from eegrep.preprocess import remove_artifact_components

    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.normal(0, 1e-5, (19, 2000)), mne.create_info(19, 128.0, "eeg"), verbose="error")
    before = raw.get_data().copy()
    log = remove_artifact_components(raw, {**cfg["preprocess"], "ica_method": "none"}, 128.0)
    assert log["n_components"] == 0 and log["excluded"] == [] and np.array_equal(raw.get_data(), before)
