"""Omnibus tests that account for the overlap between cross-validation training sets (second review, M2).

The split-level repeated-measures ANOVA and the by-subject mixed model treat the 25 test splits (or the
held-out subjects) as independent, but the models of different splits share most of their training data. The
Nadeau-Bengio correction handles this for a paired difference by inflating the variance of the mean over J
splits from s^2/J to s^2 (1/J + n_test/n_train). For a within-split factor with k levels the same inflation
applies to every contrast, so the corrected statistic is F_NB = F / (1 + J * n_test/n_train) on the same degrees
of freedom. For k = 2 this is exactly the squared Nadeau-Bengio t statistic; for k > 2 it is its natural
generalisation (each contrast's variance inflated by the same factor) and, like the original, conservative.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from .stats_grid import RATIO_TRAIN, rm_anova


def nb_adjusted_anova(df: pd.DataFrame, ratio: float = RATIO_TRAIN, dv: str = "macro_f1") -> pd.DataFrame:
    aov = rm_anova(df, dv=dv)
    j = aov["ddof2"].iloc[0] / aov["ddof1"].iloc[0] + 1          # splits rm_anova kept (it drops incomplete ones)
    inflation = 1.0 + j * ratio
    aov["nb_inflation"] = inflation
    aov["F_nb"] = aov["F"] / inflation
    aov["p_nb"] = stats.f.sf(aov["F_nb"], aov["ddof1"], aov["ddof2"])
    aov["p_nb_GG"] = stats.f.sf(aov["F_nb"], aov["ddof1"] * aov["eps-GG"], aov["ddof2"] * aov["eps-GG"])
    return aov


def write_omnibus(df: pd.DataFrame, out_dir: str | Path, prefix: str, ratio: float = RATIO_TRAIN) -> pd.DataFrame:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tab = nb_adjusted_anova(df, ratio)
    tab.to_csv(out / f"{prefix}_nb_anova.csv", index=False)
    return tab
