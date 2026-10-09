"""Subject-level repeated stratified CV with an inner validation split.

Rule 1 of the study: all windows of a subject live in exactly one of
train / val / test. `assert_disjoint` is run on every load and in the tests.
Each repeat seed defines both the partition and (in train.py) the model init.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_folds(subjects, labels, seeds, n_folds: int, val_frac: float, meta: dict | None = None) -> dict:
    subjects = np.asarray(subjects)
    labels = np.asarray(labels)
    order = np.argsort(subjects)
    subjects, labels = subjects[order], labels[order]
    repeats = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for k, (trv, te) in enumerate(skf.split(subjects, labels)):
            tr, va = train_test_split(trv, test_size=val_frac, stratify=labels[trv],
                                      random_state=seed * 1000 + k)
            folds.append({"train": sorted(subjects[tr].tolist()),
                          "val": sorted(subjects[va].tolist()),
                          "test": sorted(subjects[te].tolist())})
        repeats.append({"seed": int(seed), "folds": folds})
    folds = {"meta": {**(meta or {}), "n_subjects": len(subjects), "n_folds": n_folds,
                      "val_frac": val_frac, "seeds": [int(s) for s in seeds]},
             "repeats": repeats}
    assert_disjoint(folds, subjects)
    return folds


def assert_disjoint(folds: dict, all_subjects) -> None:
    all_subjects = set(map(str, all_subjects))
    for rep in folds["repeats"]:
        tested = []
        for k, f in enumerate(rep["folds"]):
            tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
            where = f"seed {rep['seed']} fold {k}"
            assert not (tr & va), f"train/val overlap at {where}: {sorted(tr & va)}"
            assert not (tr & te), f"train/test overlap at {where}: {sorted(tr & te)}"
            assert not (va & te), f"val/test overlap at {where}: {sorted(va & te)}"
            assert tr | va | te == all_subjects, f"fold does not cover all subjects at {where}"
            tested.extend(te)
        assert sorted(tested) == sorted(all_subjects), f"seed {rep['seed']}: each subject must be tested exactly once"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_folds(folds: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(folds, indent=1))


def load_folds(path: str | Path, all_subjects=None) -> dict:
    folds = json.loads(Path(path).read_text())
    if all_subjects is not None:
        assert_disjoint(folds, all_subjects)
    return folds


def window_level_folds(window_subjects, labels_per_window, seeds, n_folds: int, val_frac: float) -> list[dict]:
    """LEAKY protocol for ablation A10 only: windows split at random, ignoring subjects.

    Returns per-repeat lists of {"train","val","test"} *window index* arrays.
    """
    n = len(window_subjects)
    idx = np.arange(n)
    out = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for k, (trv, te) in enumerate(skf.split(idx, labels_per_window)):
            tr, va = train_test_split(trv, test_size=val_frac, stratify=labels_per_window[trv],
                                      random_state=seed * 1000 + k)
            folds.append({"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)})
        out.append({"seed": int(seed), "folds": folds})
    return out
