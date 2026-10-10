"""One training run: fold-only scaling -> weighted training -> early stopping on
validation subject-level macro-F1 -> subject-level (and window-level) test metrics.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import classification_metrics
from .models import build_model, n_params


def fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over training windows and nodes only (no leakage)."""
    flat = x_train.reshape(-1, x_train.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6


def residualize(X: np.ndarray, covariates: np.ndarray, idx_train: np.ndarray) -> np.ndarray:
    """Remove the linear effect of per-window covariates (e.g. age, sex) from every feature;
    coefficients are estimated on training windows only."""
    z = np.column_stack([np.ones(len(X)), covariates]).astype(np.float64)
    flat = X.reshape(len(X), -1).astype(np.float64)
    beta, *_ = np.linalg.lstsq(z[idx_train], flat[idx_train], rcond=None)
    return (flat - z @ beta).reshape(X.shape).astype(np.float32)


def window_weights(subjects: np.ndarray, labels: np.ndarray, n_classes: int, subject_equal: bool) -> np.ndarray:
    """Inverse class frequency over *subjects*; optionally 1/n_windows(subject) so each subject counts equally."""
    uniq, first = np.unique(subjects, return_index=True)
    subj_labels = labels[first]
    counts = np.bincount(subj_labels, minlength=n_classes).astype(float)
    class_w = len(uniq) / (n_classes * np.maximum(counts, 1))
    w = class_w[labels]
    if subject_equal:
        _, inv, n_win = np.unique(subjects, return_inverse=True, return_counts=True)
        w = w / n_win[inv]
    return (w / w.mean()).astype(np.float32)


def aggregate_by_subject(logits: np.ndarray, subjects: np.ndarray, labels: np.ndarray):
    """Subject prediction = mean of window logits."""
    uniq, inv = np.unique(subjects, return_inverse=True)
    sums = np.zeros((len(uniq), logits.shape[1]))
    np.add.at(sums, inv, logits)
    mean_logits = sums / np.bincount(inv)[:, None]
    y = np.zeros(len(uniq), dtype=int)
    y[inv] = labels
    return uniq, y, mean_logits


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_one(X: np.ndarray, A: np.ndarray, win_subject: np.ndarray, win_label: np.ndarray,
              idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray,
              cfg: dict, seed: int, device: str = "cpu", n_classes: int = 3,
              covariates: np.ndarray | None = None, refit_epochs: int | None = None) -> dict:
    """refit_epochs: if given, train on train ∪ val for exactly that many epochs (the epoch count chosen by
    early stopping on the same split) with no validation-based selection — so the GCN sees the same subjects
    as the train+val classical baselines."""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tc = cfg["train"]
    if refit_epochs is not None:
        idx_train = np.concatenate([idx_train, idx_val])

    if covariates is not None:      # ablation A12: regress age/sex out of every feature (train fit only)
        X = residualize(X, covariates, idx_train)
    mu, sd = fit_scaler(X[idx_train])
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=device)
    At = torch.as_tensor(A, dtype=torch.float32, device=device)
    shared_adj = At.dim() == 2
    yt = torch.as_tensor(win_label, dtype=torch.long, device=device)
    wt = torch.zeros(len(X), device=device)
    wt[torch.as_tensor(idx_train, device=device)] = torch.as_tensor(
        window_weights(win_subject[idx_train], win_label[idx_train], n_classes, tc["subject_equal_weighting"]),
        device=device)

    model = build_model(X.shape[-1], cfg["model"], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    def adj(idx):
        return At if shared_adj else At[idx]

    @torch.no_grad()
    def predict(idx: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        for s in range(0, len(idx), 1024):
            b = torch.as_tensor(idx[s:s + 1024], device=device)
            out.append(model(Xt[b], adj(b)).cpu().numpy())
        return np.concatenate(out)

    def subject_score(idx):
        _, y, logits = aggregate_by_subject(predict(idx), win_subject[idx], win_label[idx])
        prob = _softmax(logits)
        f1 = classification_metrics(y, prob, n_classes)["macro_f1"]
        loss = float(-np.log(prob[np.arange(len(y)), y] + 1e-12).mean())
        return f1, loss

    train_idx = torch.as_tensor(idx_train, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best = (-1.0, np.inf)
    best_state, best_epoch, epoch = None, 0, 0
    for epoch in range(1, (refit_epochs or tc["max_epochs"]) + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).to(device)]
        for s in range(0, len(perm), tc["batch_size"]):
            b = perm[s:s + tc["batch_size"]]
            if len(b) < 2:          # BatchNorm needs >1 sample
                continue
            loss = (F.cross_entropy(model(Xt[b], adj(b)), yt[b], reduction="none") * wt[b]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if refit_epochs is not None:
            continue
        f1, vloss = subject_score(idx_val)
        if f1 > best[0] or (f1 == best[0] and vloss < best[1]):
            best, best_epoch = (f1, vloss), epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= tc["patience"]:
            break

    if refit_epochs is not None:
        best_epoch = epoch                      # final weights; validation subjects were trained on
    else:
        model.load_state_dict(best_state)
    test_logits = predict(idx_test)
    subjects, y_subj, subj_logits = aggregate_by_subject(test_logits, win_subject[idx_test], win_label[idx_test])
    subj_prob = _softmax(subj_logits)
    return {
        "subject": classification_metrics(y_subj, subj_prob, n_classes),
        "window": classification_metrics(win_label[idx_test], _softmax(test_logits), n_classes),
        "val_macro_f1": best[0],
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "n_params": n_params(model),
        "seconds": time.time() - t0,
        "device": device,
        "preds": {"subjects": subjects.tolist(), "y": y_subj.tolist(), "prob": np.round(subj_prob, 5).tolist()},
    }
