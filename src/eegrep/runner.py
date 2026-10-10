"""Grid execution: resumable, budgeted, one JSON line per run.

Run key = tag | pipeline | edge | seed | fold | config_hash. Finished keys are
skipped, so a Kaggle session that hits its time limit loses nothing.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .cache import FeatureCache
from .config import apply_overrides, config_hash
from .train import train_one


class ResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.done = {json.loads(line)["key"] for line in f if line.strip()}

    def append(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.done.add(row["key"])


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_cell(cache: FeatureCache, folds: dict, cfg: dict, store: ResultStore, *, pipeline: str,
             edge: str, tag: str = "main", feature_set: list[str] | None = None, deadline: float = float("inf"),
             device: str = "cpu", max_repeats: int | None = None, overrides: dict | None = None,
             covariates: np.ndarray | None = None, labels: np.ndarray | None = None,
             window_folds: list[dict] | None = None, refit_epochs: dict | None = None) -> int:
    """Train every (repeat, fold) of one cell. Returns the number of runs executed now.

    covariates: per-window (N, d) demographics to regress out (A12).
    labels: per-window label override (permutation test).
    window_folds: explicit window-index splits (A10 leakage protocol) instead of subject folds.
    refit_epochs: {(pipeline, edge, seed, fold): epochs} — refit on train ∪ val for the early-stopped
        epoch count of the matching main-grid run (fair comparison with train+val baselines).
    """
    cfg = apply_overrides(cfg, {"graph.edge": edge, **(overrides or {})})
    chash = config_hash(cfg)
    X = cache.X_concat(feature_set) if feature_set else cache.X(pipeline)
    y = cache.label if labels is None else labels
    executed = 0
    repeats = window_folds if window_folds is not None else folds["repeats"]
    for rep in repeats[:max_repeats]:
        seed = rep["seed"]
        A = cache.adjacency(cfg, seed=seed)
        for k, f in enumerate(rep["folds"]):
            key = f"{tag}|{pipeline}|{edge}|{seed}|{k}|{chash}"
            if key in store.done:
                continue
            if time.time() > deadline:
                print(f"[runner] budget reached before {key}; stopping cleanly", flush=True)
                return executed
            if window_folds is not None:
                idx = (f["train"], f["val"], f["test"])
            else:
                idx = tuple(cache.window_indices(f[s]) for s in ("train", "val", "test"))
            n_ep = refit_epochs[(pipeline, edge, seed, k)] if refit_epochs is not None else None
            res = train_one(X, A, cache.subject, y, *idx, cfg, seed=seed * 100 + k, device=device,
                            covariates=covariates, refit_epochs=n_ep)
            store.append({"key": key, "tag": tag, "pipeline": pipeline, "edge": edge, "seed": seed, "fold": k,
                          "config_hash": chash, "overrides": overrides or {}, "git_sha": git_sha(),
                          "python": platform.python_version(), **res})
            executed += 1
            print(f"[runner] {key}  macroF1={res['subject']['macro_f1']:.3f}  "
                  f"epochs={res['epochs_run']}  {res['seconds']:.1f}s", flush=True)
    return executed


def bench(cache: FeatureCache, folds: dict, cfg: dict, *, pipeline: str, edge: str, n_runs_planned: int,
          max_hours: float, device: str = "cpu", out: str | Path | None = None) -> float:
    """Time one repeat (all folds) of a cell, project the cost of `n_runs_planned` runs.

    Raises SystemExit if the projection exceeds `max_hours`.
    """
    store = ResultStore(out or Path("results") / "bench.jsonl")
    before = len(store.done)
    t0 = time.time()
    run_cell(cache, folds, cfg, store, pipeline=pipeline, edge=edge, tag="bench", device=device, max_repeats=1)
    n = len(store.done) - before
    if n == 0:
        raise RuntimeError("bench already complete in this store; delete it or pass a fresh --out")
    per_run = (time.time() - t0) / n
    hours = per_run * n_runs_planned / 3600
    print(f"[bench] {per_run:.1f} s/run on {device} -> {n_runs_planned} runs ≈ {hours:.2f} h "
          f"(budget {max_hours} h)", flush=True)
    if hours > max_hours:
        raise SystemExit(f"projected {hours:.2f} h exceeds budget {max_hours} h — shard the grid")
    return hours


def best_epochs(run_files, tag: str = "main") -> dict:
    """{(pipeline, edge, seed, fold): best_epoch} from earlier early-stopped runs."""
    out = {}
    for p in run_files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["tag"] == tag:
                        out[(r["pipeline"], r["edge"], r["seed"], r["fold"])] = max(1, int(r["best_epoch"]))
    return out


def summarize(path: str | Path) -> dict:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["tag"], r["pipeline"], r["edge"]), []).append(r["subject"]["macro_f1"])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in by_cell.items()}
