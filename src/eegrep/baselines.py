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
                met = classification_metrics(y_all.loc[f["test"]].values, prob, y_all.nunique())
                rows.append({"model": name, "seed": rep["seed"], "fold": k,
                             **{key: v for key, v in met.items() if key != "confusion"}})
    return pd.DataFrame(rows)


def _classical_models(seed: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    return {
        "lr": LogisticRegression(class_weight=None, max_iter=3000, C=1.0),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed),
    }


def classical_baseline(cache, folds: dict, pipeline: str, cfg: dict, fit_on: str = "trainval") -> pd.DataFrame:
    """Non-graph twin of each representation: flattened (C·F) window features, fixed default
    hyperparameters, same subject-equal × inverse-class weights as the GCN, subject prediction
    = mean of window log-probabilities. Scaler fit on training windows only."""
    from .train import aggregate_by_subject, window_weights

    X = cache.X(pipeline).reshape(len(cache.subject), -1)
    rows = []
    for rep in folds["repeats"]:
        for k, f in enumerate(rep["folds"]):
            # "train": the same subjects the GCN trains on (fair comparison);
            # "trainval": train + validation subjects (no tuning, so validation is otherwise unused)
            tr = cache.window_indices(f["train"] + (f["val"] if fit_on == "trainval" else []))
            te = cache.window_indices(f["test"])
            n_cls = len(cfg["dataset"]["class_names"])
            w = window_weights(cache.subject[tr], cache.label[tr], n_cls, cfg["train"]["subject_equal_weighting"])
            scaler = StandardScaler().fit(X[tr])
            for name, model in _classical_models(rep["seed"] * 100 + k).items():
                model.fit(scaler.transform(X[tr]), cache.label[tr], sample_weight=w)
                logp = np.log(model.predict_proba(scaler.transform(X[te])) + 1e-9)
                _, y, mean_logp = aggregate_by_subject(logp, cache.subject[te], cache.label[te])
                prob = np.exp(mean_logp - mean_logp.max(1, keepdims=True))
                met = classification_metrics(y, prob / prob.sum(1, keepdims=True), n_cls)
                rows.append({"model": name, "pipeline": pipeline, "fit_on": fit_on, "seed": rep["seed"], "fold": k, **met})
                print(f"[classical] {pipeline} {name} seed{rep['seed']} fold{k} macroF1={met['macro_f1']:.3f}",
                      flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str = "macro_f1", by=("model",)) -> pd.DataFrame:
    g = df.groupby(list(by))[metric]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.size(),
                         "ci95_lo": g.mean() - 1.96 * g.std() / np.sqrt(g.size()),
                         "ci95_hi": g.mean() + 1.96 * g.std() / np.sqrt(g.size())})
