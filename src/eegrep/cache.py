"""Feature cache: the hand-off between CPU preprocessing (NB01/NB02) and training.

Layout of `cache_dir`:
  index.npz         subject (N,), start_s (N,), label (N,), channels, coords (C, 3), sfreq
  features/P{1..7}.npz   X (N, C, F)
  wpli.npz          W (N, B, C, C) float16, band names
  pswe_jaccard.npz  subjects (S,), J (S, C, C)
  pswe_events.csv, pswe_subject.csv   (RQ6 statistics)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import connectivity, features, graphs, pswe

PIPELINES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def windowize(x: np.ndarray, sfreq: float, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(C, T) -> non-overlapping windows (N, C, W), start times (N,)."""
    w = int(round(window_s * sfreq))
    n = x.shape[-1] // w
    wins = x[:, :n * w].reshape(x.shape[0], n, w).transpose(1, 0, 2)
    return np.ascontiguousarray(wins, dtype=np.float32), np.arange(n) * window_s


def build_cache(recordings, labels: dict, cfg: dict, coords: np.ndarray, out_dir: str | Path) -> Path:
    """recordings: iterable of (subject, data (C, T) in µV, sfreq) — preprocessed, continuous.

    µV (not volts) keeps feature magnitudes far above the numerical epsilons in features.py.
    """
    out = Path(out_dir)
    (out / "features").mkdir(parents=True, exist_ok=True)
    channels = cfg["dataset"]["channels"]
    win_s = cfg["preprocess"]["window_s"]
    acc = {p: [] for p in PIPELINES}
    subj_col, start_col, wpli_all, ev_all, summ_all, jac_subj, jac = [], [], [], [], [], [], []
    sfreq_seen = None
    for subject, data, sfreq in recordings:
        sfreq_seen = sfreq
        wins, starts = windowize(data, sfreq, win_s)
        w_b = connectivity.wpli_bands(wins, sfreq, cfg["bands"])
        for name, x in features.compute_all(wins, sfreq, w_b, cfg).items():
            acc[name].append(x)
        events, mpf, mask, step = pswe.detect(data, sfreq, cfg["pswe"], channels)
        duration = data.shape[-1] / sfreq
        acc["P7"].append(pswe.window_features(mpf, mask, events, channels, starts, win_s, step,
                                              cfg["features"]["pswe_local_rate_s"], duration))
        jac_subj.append(subject)
        jac.append(graphs.jaccard_adjacency(mask))
        ev_all.append(events.assign(subject=subject))
        summ_all.append(pswe.subject_summary(events, mask, channels, duration)
                        .assign(subject=subject, duration_s=duration))
        wpli_all.append(w_b.astype(np.float16))
        subj_col += [subject] * len(wins)
        start_col.append(starts)
        print(f"[cache] {subject}: {len(wins)} windows, {len(events)} PSWE events", flush=True)

    subj_arr = np.array(subj_col)
    np.savez(out / "index.npz", subject=subj_arr, start_s=np.concatenate(start_col),
             label=np.array([labels[s] for s in subj_arr]), channels=np.array(channels),
             coords=coords, sfreq=sfreq_seen)
    for name, chunks in acc.items():
        np.savez(out / "features" / f"{name}.npz", X=np.concatenate(chunks))
    np.savez(out / "wpli.npz", W=np.concatenate(wpli_all), bands=np.array(list(cfg["bands"])))
    np.savez(out / "pswe_jaccard.npz", subjects=np.array(jac_subj), J=np.stack(jac))
    pd.concat(ev_all).to_csv(out / "pswe_events.csv", index=False)
    pd.concat(summ_all).to_csv(out / "pswe_subject.csv", index=False)
    return out


class FeatureCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        idx = np.load(self.dir / "index.npz")
        self.subject = idx["subject"].astype(str)
        self.label = idx["label"].astype(int)
        self.start_s = idx["start_s"]
        self.coords = idx["coords"]
        self._wpli = None

    def X(self, pipeline: str) -> np.ndarray:
        return np.load(self.dir / "features" / f"{pipeline}.npz")["X"]

    def X_concat(self, pipelines: list[str]) -> np.ndarray:
        return np.concatenate([self.X(p) for p in pipelines], axis=-1)

    def wpli(self, band: str) -> np.ndarray:
        if self._wpli is None:
            self._wpli = np.load(self.dir / "wpli.npz")
        bands = list(self._wpli["bands"].astype(str))
        return self._wpli["W"][:, bands.index(band)].astype(np.float32)

    def pswe_per_window(self) -> np.ndarray:
        z = np.load(self.dir / "pswe_jaccard.npz")
        pos = {s: i for i, s in enumerate(z["subjects"].astype(str))}
        return z["J"][[pos[s] for s in self.subject]].astype(np.float32)

    def adjacency(self, cfg: dict, seed: int = 0) -> np.ndarray:
        g = cfg["graph"]
        edge = g["edge"]
        spatial = graphs.spatial_adjacency(self.coords)
        functional = self.wpli(g["functional_band"]) if edge in ("functional", "hybrid", "random") else None
        pswe_adj = self.pswe_per_window() if edge == "pswe" else None
        return graphs.build(edge, spatial=spatial, functional=functional, pswe=pswe_adj,
                            k=g["knn_k"], binarize=g["binarize"], seed=seed)

    def window_indices(self, subjects) -> np.ndarray:
        return np.flatnonzero(np.isin(self.subject, list(subjects)))
