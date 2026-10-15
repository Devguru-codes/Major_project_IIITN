"""Out-of-bootstrap comparison of representations with retraining (sixth review, could item).

Cross-validation splits of one cohort share most of their training subjects, so split-level tests are either
anti-conservative (unadjusted) or very conservative (Nadeau-Bengio). The out-of-bootstrap scheme resamples the
whole cohort instead: in replicate b, subjects are drawn with replacement within each class, the GCN is trained
on the distinct drawn subjects (20% of them held out for early stopping, as in the main protocol; repeated
draws of a subject carry no extra weight because the loss is subject-equal) and scored on the subjects never
drawn. Every representation is trained and scored on the same replicate, so differences are paired. The
spread of a paired difference across replicates reflects both the training sample and the test sample; its
standard deviation is the bootstrap standard error of a paired difference (z test, Holm-corrected), and
its percentile interval is reported alongside.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from .config import apply_overrides
from .stats_pswe import holm
from .train import train_one


def bootstrap_split(subjects: np.ndarray, labels: np.ndarray, b: int, val_frac: float = 0.2):
    rng = np.random.default_rng(10_000 + b)
    drawn = np.concatenate([rng.choice(subjects[labels == c], size=(labels == c).sum(), replace=True)
                            for c in np.unique(labels)])
    inbag = np.unique(drawn)
    oob = np.setdiff1d(subjects, inbag)
    lab = dict(zip(subjects, labels))
    y_in = [lab[s] for s in inbag]
    can_stratify = min(np.unique(y_in, return_counts=True)[1]) >= 2     # always true on the real cohort
    tr, va = train_test_split(inbag, test_size=val_frac, stratify=y_in if can_stratify else None, random_state=b)
    return sorted(tr), sorted(va), sorted(oob)


def run_bootstrap(cache, cfg, pipelines: list[str], edge: str, b_values, out_path: str | Path) -> int:
    cfg = apply_overrides(cfg, {"graph.edge": edge})
    subjects = np.array(sorted(set(cache.subject)))
    labels = np.array([int(cache.label[cache.subject == s][0]) for s in subjects])
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(r["b"], r["pipeline"]) for r in map(json.loads, out_path.read_text().splitlines()) if r}
    A = cache.adjacency(cfg, seed=0)
    n = 0
    for b in b_values:
        tr, va, oob = bootstrap_split(subjects, labels, b)
        idx = tuple(cache.window_indices(s) for s in (tr, va, oob))
        for p in pipelines:
            if (b, p) in done:
                continue
            res = train_one(cache.X(p), A, cache.subject, cache.label, *idx, cfg, seed=20_000 + b,
                            n_classes=len(cfg["dataset"]["class_names"]))
            row = {"b": int(b), "pipeline": p, "edge": edge, "n_train": len(tr), "n_val": len(va), "n_oob": len(oob),
                   "macro_f1": res["subject"]["macro_f1"], "balanced_acc": res["subject"]["balanced_acc"],
                   "seconds": res["seconds"]}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            n += 1
            print(f"[bootstrap] b={b} {p} macroF1={row['macro_f1']:.3f} {res['seconds']:.1f}s", flush=True)
    return n


def summarise(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per representation: mean out-of-bootstrap macro-F1 with percentile interval; every pair: paired
    difference, 95% percentile interval and two-sided bootstrap p (Holm across the pairs)."""
    wide = rows.pivot(index="b", columns="pipeline", values="macro_f1").dropna()
    reps = wide.agg(["mean", lambda s: s.quantile(0.025), lambda s: s.quantile(0.975)]).T
    reps.columns = ["mean", "lo", "hi"]
    reps["n_boot"] = len(wide)
    pairs = []
    cols = list(wide.columns)
    for i, a in enumerate(cols):
        for c in cols[i + 1:]:
            d = (wide[a] - wide[c]).to_numpy()
            se = d.std(ddof=1)
            pairs.append({"a": a, "b": c, "delta": d.mean(), "se_boot": se, "lo": np.quantile(d, 0.025),
                          "hi": np.quantile(d, 0.975),
                          "p_boot": float(2 * stats.norm.sf(abs(d.mean()) / se)) if se > 0 else 1.0,
                          "p_percentile": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = holm(pairs["p_boot"].tolist())
    return reps.reset_index(), pairs
