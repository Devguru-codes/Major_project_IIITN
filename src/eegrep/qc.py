"""Data-quality gates for the merged cache (IMPLEMENTATION_PLAN.md §8, phases P1–P2).

Expected from the dataset's own files: 88 subjects (36 AD / 23 FTD / 29 CN),
per-subject duration 307–1291 s, ≈7,059 ten-second windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from .cache import PIPELINES, FeatureCache


def qc_report(cache_dir: str | Path, preprocess_logs: list[str | Path], class_names: list[str]) -> dict:
    cache = FeatureCache(cache_dir)
    subj, lab = cache.subject, cache.label
    uniq, first, n_win = np.unique(subj, return_index=True, return_counts=True)
    subj_lab = lab[first]
    rep = {"n_subjects": int(len(uniq)),
           "class_counts": {class_names[c]: int((subj_lab == c).sum()) for c in range(len(class_names))},
           "n_windows": int(len(subj)),
           "windows_per_subject": {"min": int(n_win.min()), "max": int(n_win.max()), "mean": float(n_win.mean())}}

    logs = pd.DataFrame([json.loads(l) for p in preprocess_logs for l in open(p, encoding="utf-8") if l.strip()])
    if not logs.empty:
        n_ex = logs["excluded"].apply(len)
        rep["duration_s"] = {"min": float(logs.duration_s.min()), "max": float(logs.duration_s.max()),
                             "mean": float(logs.duration_s.mean())}
        rep["ica_excluded"] = {"mean": float(n_ex.mean()), "min": int(n_ex.min()), "max": int(n_ex.max()),
                               "labels": pd.Series([x for xs in logs.excluded_labels for x in xs])
                               .value_counts().to_dict()}

    rep["finite"] = {p: bool(np.isfinite(cache.X(p)).all()) for p in PIPELINES}

    # Sanity: EEG slowing — theta/alpha relative power per subject (P1 cols 1, 2)
    p1 = cache.X("P1")
    ta = pd.Series(p1[..., 1].mean(-1) / p1[..., 2].mean(-1)).groupby(subj).mean().loc[uniq].values
    rep["theta_alpha_ratio"] = {class_names[c]: float(np.median(ta[subj_lab == c])) for c in range(len(class_names))}
    rep["theta_alpha_AD_gt_CN_p"] = float(mannwhitneyu(ta[subj_lab == 0], ta[subj_lab == 2],
                                                       alternative="greater").pvalue)

    # PSWE burden per group (descriptive; inferential stats live in NB03b)
    ps = pd.read_csv(Path(cache_dir) / "pswe_subject.csv").groupby("subject")
    rate = ps["rate_per_min"].mean().loc[uniq].values
    frac = ps["time_fraction"].mean().loc[uniq].values
    rep["pswe_rate_per_min_median"] = {class_names[c]: float(np.median(rate[subj_lab == c]))
                                       for c in range(len(class_names))}
    rep["pswe_time_fraction_median"] = {class_names[c]: float(np.median(frac[subj_lab == c]))
                                        for c in range(len(class_names))}
    rep["pswe_rate_kruskal_p"] = float(kruskal(*[rate[subj_lab == c] for c in range(len(class_names))]).pvalue)
    return rep


def gate_failures(rep: dict) -> list[str]:
    fails = []
    if rep["n_subjects"] != 88:
        fails.append(f"expected 88 subjects, got {rep['n_subjects']}")
    if not 6500 <= rep["n_windows"] <= 7100:
        fails.append(f"window count {rep['n_windows']} outside 6500–7100 (expected ≈7,059)")
    if "duration_s" in rep and not (300 <= rep["duration_s"]["min"] and rep["duration_s"]["max"] <= 1300):
        fails.append(f"durations {rep['duration_s']} outside 307–1291 s")
    if not all(rep["finite"].values()):
        fails.append(f"non-finite features: {rep['finite']}")
    if rep["theta_alpha_ratio"]["AD"] <= rep["theta_alpha_ratio"]["CN"]:
        fails.append("sanity: AD theta/alpha not above CN (known EEG slowing)")
    return fails
