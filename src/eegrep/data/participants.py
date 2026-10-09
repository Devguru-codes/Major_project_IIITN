"""participants.tsv loading with the MMSE firewall.

Rule 2 of the study: MMSE is a label proxy (CN = 30.0 ± 0.0) and must never
reach a model. It is kept only in `analysis_table` for RQ6 correlations.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FORBIDDEN_MODEL_COLUMNS = frozenset({"mmse", "MMSE"})


def load_participants(path: str | Path, group_to_label: dict) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"participant_id": "subject", "Gender": "sex", "Age": "age", "Group": "group", "MMSE": "mmse"})
    unknown = set(df["group"]) - set(group_to_label)
    if unknown:
        raise ValueError(f"unknown group codes in participants.tsv: {unknown}")
    df["label"] = df["group"].map(group_to_label).astype(int)
    df["sex_male"] = (df["sex"] == "M").astype(int)
    return df.sort_values("subject").reset_index(drop=True)


def model_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """The only non-EEG inputs any model may see (demographics baseline)."""
    out = df[["subject", "age", "sex_male"]].copy()
    assert_no_mmse(out.columns)
    return out


def analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    """Statistics-only table (includes MMSE). Never pass to a model."""
    return df[["subject", "group", "label", "age", "sex_male", "mmse"]].copy()


def assert_no_mmse(columns) -> None:
    bad = FORBIDDEN_MODEL_COLUMNS.intersection(map(str, columns))
    if bad:
        raise AssertionError(f"MMSE must never be a model input; found {sorted(bad)}")
