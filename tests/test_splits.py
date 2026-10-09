import json

import numpy as np
import pytest

from eegrep.splits import assert_disjoint, load_folds, make_folds, save_folds, window_level_folds

# ds004504 composition: 36 AD / 23 FTD / 29 CN
SUBJECTS = [f"sub-{i:03d}" for i in range(1, 89)]
LABELS = [0] * 36 + [1] * 23 + [2] * 29


@pytest.fixture(scope="module")
def folds():
    return make_folds(SUBJECTS, LABELS, seeds=[0, 1, 2, 3, 4], n_folds=5, val_frac=0.2)


def test_train_val_test_disjoint_everywhere(folds):
    for rep in folds["repeats"]:
        for f in rep["folds"]:
            tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
            assert not (tr & va) and not (tr & te) and not (va & te)
            assert tr | va | te == set(SUBJECTS)


def test_each_subject_tested_once_per_repeat(folds):
    for rep in folds["repeats"]:
        tested = sorted(s for f in rep["folds"] for s in f["test"])
        assert tested == sorted(SUBJECTS)


def test_test_folds_are_stratified(folds):
    lab = dict(zip(SUBJECTS, LABELS))
    for rep in folds["repeats"]:
        for f in rep["folds"]:
            counts = np.bincount([lab[s] for s in f["test"]], minlength=3)
            assert np.all(np.abs(counts - np.array([36, 23, 29]) / 5) <= 1)


def test_repeats_differ(folds):
    assert folds["repeats"][0]["folds"][0]["test"] != folds["repeats"][1]["folds"][0]["test"]


def test_json_roundtrip_and_validation(folds, tmp_path):
    p = tmp_path / "folds.json"
    save_folds(folds, p)
    assert load_folds(p, SUBJECTS) == json.loads(p.read_text())


def test_tampered_folds_are_rejected(folds):
    bad = json.loads(json.dumps(folds))
    f = bad["repeats"][0]["folds"][0]
    f["train"].append(f["test"][0])
    with pytest.raises(AssertionError):
        assert_disjoint(bad, SUBJECTS)


def test_window_level_folds_leak_subjects_by_design():
    win_subj = np.repeat(SUBJECTS, 10)
    win_lab = np.repeat(LABELS, 10)
    rep = window_level_folds(win_subj, win_lab, seeds=[0], n_folds=5, val_frac=0.2)[0]
    f = rep["folds"][0]
    assert set(win_subj[f["train"]]) & set(win_subj[f["test"]]), "leaky protocol should share subjects"
