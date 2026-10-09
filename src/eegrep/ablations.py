"""Ablations (IMPLEMENTATION_PLAN.md §5), run on the best cell(s) of the main grid with the
same frozen 5×5 folds, so every ablation is paired with its main-grid cell.

A7 (window length) and A8 (ICA off) need a re-built cache and run via separate NB01 variants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache import FeatureCache
from .runner import ResultStore, run_cell
from .splits import window_level_folds

# name -> how it differs from the reference cell
ABLATIONS = {
    "A1_no_graph": {"edge": "identity"},
    "A2_random_graph": {"edge": "random"},
    "A3_k3": {"overrides": {"graph.knn_k": 3}},
    "A3_k6": {"overrides": {"graph.knn_k": 6}},
    "A3_full": {"overrides": {"graph.knn_k": None}},
    "A4_binary": {"overrides": {"graph.binarize": True}},
    "A5_depth1": {"overrides": {"model.hidden": [32]}},
    "A5_depth3": {"overrides": {"model.hidden": [64, 64, 32]}},
    "A6_gat": {"overrides": {"model.arch": "gat"}},
    "A9_unweighted": {"overrides": {"train.subject_equal_weighting": False}},
    "A12_residualized": {"covariates": True},
    "RQ7_plus_P7": {"add_p7": True},
    "RQ7_pswe_edge": {"edge": "pswe"},
    "A10_window_split": {"leaky": True},
}


def window_covariates(cache: FeatureCache, participants: pd.DataFrame) -> np.ndarray:
    p = participants.set_index("subject")
    return np.column_stack([p.loc[cache.subject, "age"].to_numpy(), p.loc[cache.subject, "sex_male"].to_numpy()])


def run_ablation(name: str, cache: FeatureCache, folds: dict, cfg: dict, store: ResultStore, *, pipeline: str,
                 edge: str, participants: pd.DataFrame | None = None, deadline: float = float("inf"),
                 device: str = "cpu") -> int:
    spec = ABLATIONS[name]
    kw = {"overrides": spec.get("overrides"), "deadline": deadline, "device": device,
          "tag": f"{name}@{pipeline}x{edge}"}
    if spec.get("covariates"):
        kw["covariates"] = window_covariates(cache, participants)
    if spec.get("add_p7"):
        kw["feature_set"] = [pipeline, "P7"]
    if spec.get("leaky"):
        kw["window_folds"] = window_level_folds(cache.subject, cache.label, folds["meta"]["seeds"],
                                                folds["meta"]["n_folds"], folds["meta"]["val_frac"])
    return run_cell(cache, folds, cfg, store, pipeline=pipeline, edge=spec.get("edge", edge), **kw)


def select_cells(run_files, top: int = 2) -> list[str]:
    """Top-`top` main-grid cells by mean subject-level macro-F1 over all completed runs."""
    import json

    rows = [json.loads(l) for f in run_files for l in open(f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame([{"cell": f"{r['pipeline']}x{r['edge']}", "f1": r["subject"]["macro_f1"]}
                       for r in rows if r["tag"] == "main"])
    means = df.groupby("cell")["f1"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    print(means.round(4).to_string())
    return list(means.index[:top])


def run_permutations(cache: FeatureCache, folds: dict, cfg: dict, store: ResultStore, *, pipeline: str, edge: str,
                     n_perm: int, perm_start: int = 0, deadline: float = float("inf"), device: str = "cpu") -> int:
    """Label-permutation null for the best cell: subject labels shuffled (all windows of a subject keep
    one label), first repeat's 5 folds per permutation. p = (1 + #null ≥ observed) / (1 + n_perm).
    Permutations [perm_start, perm_start + n_perm) so the null can be split across kernels."""
    subjects, first = np.unique(cache.subject, return_index=True)
    true = cache.label[first]
    pos = np.searchsorted(subjects, cache.subject)
    done = 0
    for p in range(perm_start, perm_start + n_perm):
        perm_lab = np.random.default_rng(10_000 + p).permutation(true)
        sub_folds = {"meta": folds["meta"], "repeats": [{**folds["repeats"][0], "seed": 10_000 + p}]}
        done += run_cell(cache, sub_folds, cfg, store, pipeline=pipeline, edge=edge, tag=f"perm@{pipeline}x{edge}",
                         labels=perm_lab[pos], deadline=deadline, device=device)
    return done
