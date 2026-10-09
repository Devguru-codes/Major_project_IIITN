"""Non-graph baselines evaluated on the same frozen subject-level folds.

Demographics-only (age + sex) is the confound check and is run first: if it is
far above chance, group differences in age/sex could explain any EEG result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data.participants import assert_no_mmse, model_covariates
from .metrics import classification_metrics


def demographics_baseline(participants: pd.DataFrame, folds: dict) -> pd.DataFrame:
    """Logistic regression on age + sex vs a stratified dummy, per repeat x fold.

    No hyperparameters are tuned, so train and val subjects are both used for fitting.
    """
    cov = model_covariates(participants).set_index("subject")
    assert_no_mmse(cov.columns)
    y_all = participants.set_index("subject")["label"]
    rows = []
    for rep in folds["repeats"]:
        for k, f in enumerate(rep["folds"]):
            fit = f["train"] + f["val"]
            models = {
                "demographics_lr": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                                                                      max_iter=2000)),
                "chance_stratified": DummyClassifier(strategy="stratified", random_state=rep["seed"] * 100 + k),
            }
            for name, m in models.items():
                m.fit(cov.loc[fit].values, y_all.loc[fit].values)
                prob = m.predict_proba(cov.loc[f["test"]].values)
                met = classification_metrics(y_all.loc[f["test"]].values, prob)
                rows.append({"model": name, "seed": rep["seed"], "fold": k,
                             **{key: v for key, v in met.items() if key != "confusion"}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    g = df.groupby("model")[metric]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.size(),
                         "ci95_lo": g.mean() - 1.96 * g.std() / np.sqrt(g.size()),
                         "ci95_hi": g.mean() + 1.96 * g.std() / np.sqrt(g.size())})
