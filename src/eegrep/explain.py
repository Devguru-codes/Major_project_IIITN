"""Which electrodes and frequency bands does the best configuration use? (Should-fix after internal review.)

Occlusion on held-out subjects: the best cell is retrained on every split with the main-grid seeds; on that split's
test windows one electrode (all its features), or one wavelet sub-band (its four features on every electrode),
is replaced by its training mean (zero after standardisation), and the drop in the probability assigned to each
test subject's true class is recorded. Importances are averaged over the five repeats per subject, then over
subjects (overall and per true class). Model-agnostic, needs no gradients, and evaluated only on unseen
subjects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import apply_overrides
from .train import _softmax, aggregate_by_subject, train_one

P3_BANDS = {"A5 (0-2 Hz)": 0, "D5 (2-4 Hz)": 1, "D4 (4-8 Hz)": 2, "D3 (8-16 Hz)": 3, "D2 (16-32 Hz)": 4}


@torch.no_grad()
def _subject_probs(model, X, A, idx, subjects, labels):
    model.eval()
    out = []
    for s in range(0, len(idx), 1024):
        b = idx[s:s + 1024]
        a = A if A.dim() == 2 else A[torch.as_tensor(b)]
        out.append(model(X[b], a).numpy())
    subj, y, logits = aggregate_by_subject(np.concatenate(out), subjects[idx], labels[idx])
    return subj, y, _softmax(logits)


def occlusion(cache, folds, cfg, pipeline: str = "P3", edge: str = "spatial", max_repeats: int | None = None,
              band_groups: dict | None = None) -> pd.DataFrame:
    """Rows: (seed, fold, subject, true class, unit kind, unit name, drop in p_true)."""
    cfg = apply_overrides(cfg, {"graph.edge": edge})
    X = cache.X(pipeline)
    n_feat = X.shape[-1]
    groups = band_groups if band_groups is not None else (
        {k: list(range(4 * v, 4 * v + 4)) for k, v in P3_BANDS.items()} if pipeline == "P3" else {})
    channels = list(cfg["dataset"]["channel_rename"].get(c, c) for c in cfg["dataset"]["channels"])
    rows = []
    for rep in folds["repeats"][:max_repeats]:
        seed = rep["seed"]
        A = torch.as_tensor(cache.adjacency(cfg, seed=seed), dtype=torch.float32)
        for k, f in enumerate(rep["folds"]):
            idx = tuple(cache.window_indices(f[s]) for s in ("train", "val", "test"))
            res = train_one(X, A.numpy(), cache.subject, cache.label, *idx, cfg, seed=seed * 100 + k,
                            n_classes=len(cfg["dataset"]["class_names"]), keep_model=True)
            model, (mu, sd) = res["_model"].cpu(), res["_scale"]
            Xs = torch.as_tensor((X - mu) / sd, dtype=torch.float32)
            test = idx[2]
            subj, y, base = _subject_probs(model, Xs, A, test, cache.subject, cache.label)
            p0 = base[np.arange(len(y)), y]

            def record(kind, name, Xo):
                _, _, p = _subject_probs(model, Xo, A, test, cache.subject, cache.label)
                for s_, y_, d_ in zip(subj, y, p0 - p[np.arange(len(y)), y]):
                    rows.append((seed, k, s_, int(y_), kind, name, float(d_)))

            for c, ch in enumerate(channels):
                Xo = Xs.clone(); Xo[:, c, :] = 0.0
                record("electrode", ch, Xo)
            for name, cols in groups.items():
                Xo = Xs.clone(); Xo[:, :, cols] = 0.0
                record("band", name, Xo)
            assert n_feat >= max((max(v) for v in groups.values()), default=0)
    return pd.DataFrame(rows, columns=["seed", "fold", "subject", "true_class", "kind", "unit", "drop"])


def summarise(occ: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Mean drop per unit, overall and per true class, with a subject-bootstrap 95% CI (overall)."""
    per_subj = occ.groupby(["kind", "unit", "subject", "true_class"], as_index=False)["drop"].mean()
    rng = np.random.default_rng(0)
    rows = []
    for (kind, unit), g in per_subj.groupby(["kind", "unit"]):
        v = g["drop"].to_numpy()
        boot = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        row = {"kind": kind, "unit": unit, "drop": v.mean(), "ci_lo": float(np.percentile(boot, 2.5)),
               "ci_hi": float(np.percentile(boot, 97.5))}
        for c, name in enumerate(class_names):
            row[f"drop_{name}"] = g.loc[g.true_class == c, "drop"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["kind", "drop"], ascending=[True, False])


def write_explain(cache, folds, cfg, out_dir, pipeline="P3", edge="spatial") -> pd.DataFrame:
    from . import figures as F

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    occ = occlusion(cache, folds, cfg, pipeline, edge)
    occ.to_csv(out / "occlusion_raw.csv", index=False)
    s = summarise(occ, cfg["dataset"]["class_names"])
    s.to_csv(out / "occlusion_summary.csv", index=False)
    F.fig_importance(s, cfg, out / "figures", title=f"{pipeline} x {edge}")
    return s
