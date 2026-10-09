"""Imbalance-aware metrics. Macro-F1 is primary; accuracy is reported, never ranked on."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    conf, pred = prob.max(1), prob.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, prob: np.ndarray, n_classes: int = 3) -> dict:
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    pred = prob.argmax(1)
    labels = list(range(n_classes))
    cm = confusion_matrix(y, pred, labels=labels)
    onehot = np.eye(n_classes)[y]
    out = {
        "macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": float((pred == y).mean()),
        "kappa": cohen_kappa_score(y, pred, labels=labels),
        "brier": float(((prob - onehot) ** 2).sum(1).mean()),
        "ece": expected_calibration_error(y, prob),
        "confusion": cm.tolist(),
    }
    present = np.unique(y)
    if len(present) == n_classes:
        out["roc_auc"] = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=labels)
        out["pr_auc"] = float(np.mean([average_precision_score(onehot[:, c], prob[:, c]) for c in labels]))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    for c in labels:
        tp, fn = cm[c, c], cm[c].sum() - cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - tp - fn - fp
        out[f"sens_{c}"] = tp / (tp + fn) if tp + fn else float("nan")
        out[f"spec_{c}"] = tn / (tn + fp) if tn + fp else float("nan")
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in out.items()}
