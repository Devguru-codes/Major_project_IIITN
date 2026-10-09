"""Factor B: graph constructions over the 19 electrodes.

All builders return raw weighted adjacency (zero diagonal); `prepare` then
applies kNN sparsification (k per row, symmetrised A = max(A, Aᵀ)), optional
binarisation, and GCN normalisation  Â = D̃^-½ (A + I) D̃^-½.
"""
from __future__ import annotations

import numpy as np


def electrode_positions(channels: list[str], rename: dict) -> np.ndarray:
    """(C, 3) head coordinates from MNE's standard_1020 montage."""
    import mne

    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    return np.array([pos[rename.get(ch, ch)] for ch in channels], dtype=np.float64)


def spatial_adjacency(coords: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Gaussian kernel on Euclidean distance; σ defaults to the median pairwise distance."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    if sigma is None:
        sigma = float(np.median(d[np.triu_indices_from(d, k=1)]))
    w = np.exp(-(d ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(w, 0.0)
    return w.astype(np.float32)


def hybrid_adjacency(spatial: np.ndarray, functional: np.ndarray) -> np.ndarray:
    return (spatial[None] * functional).astype(np.float32)


def jaccard_adjacency(masks: np.ndarray) -> np.ndarray:
    """masks: (C, S) boolean in-event masks -> (C, C) Jaccard overlap (PSWE co-occurrence)."""
    m = masks.astype(np.float32)
    inter = m @ m.T
    count = m.sum(1)
    union = count[:, None] + count[None, :] - inter
    j = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(j, 0.0)
    return j


def knn_sparsify(a: np.ndarray, k: int | None) -> np.ndarray:
    """Keep the k strongest off-diagonal edges per row, then symmetrise. k=None keeps all."""
    a = np.asarray(a, dtype=np.float32)
    if k is None:
        return a
    c = a.shape[-1]
    masked = a.copy()
    idx = np.arange(c)
    masked[..., idx, idx] = -np.inf
    top = np.argpartition(-masked, kth=k - 1, axis=-1)[..., :k]
    keep = np.zeros(a.shape, dtype=bool)
    np.put_along_axis(keep, top, True, axis=-1)
    sparse = np.where(keep, a, 0.0)
    return np.maximum(sparse, np.swapaxes(sparse, -1, -2)).astype(np.float32)


def normalize(a: np.ndarray) -> np.ndarray:
    """Â = D̃^-½ (A + I) D̃^-½ for (..., C, C)."""
    c = a.shape[-1]
    at = a + np.eye(c, dtype=np.float32)
    dinv = 1.0 / np.sqrt(at.sum(-1))
    return (dinv[..., :, None] * at * dinv[..., None, :]).astype(np.float32)


def degree_preserving_shuffle(a: np.ndarray, seed: int) -> np.ndarray:
    """Ablation A2: relabel nodes at random per graph (same degree multiset, different topology)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    single = a.ndim == 2
    batch = a[None] if single else a
    out = np.empty_like(batch)
    for i, g in enumerate(batch):
        p = rng.permutation(g.shape[-1])
        out[i] = g[np.ix_(p, p)]
    return out[0] if single else out


def prepare(raw: np.ndarray, k: int | None, binarize: bool = False) -> np.ndarray:
    a = knn_sparsify(raw, k)
    if binarize:
        a = (a > 0).astype(np.float32)
    return normalize(a)


def build(edge: str, *, spatial: np.ndarray, functional: np.ndarray | None = None,
          pswe: np.ndarray | None = None, k: int | None = 4, binarize: bool = False,
          seed: int = 0) -> np.ndarray:
    """Return normalised adjacency: (C, C) if shared by all windows, else (N, C, C).

    edge: spatial | functional | hybrid | pswe | identity | random
      functional: (N, C, C) per-window α-wPLI;  pswe: (N, C, C) per-window copy of the
      recording's Jaccard graph;  random: degree-preserving shuffle of the hybrid graph.
    """
    c = spatial.shape[-1]
    if edge == "identity":
        return np.eye(c, dtype=np.float32)
    if edge == "spatial":
        return prepare(spatial, k, binarize)
    if edge == "functional":
        return prepare(functional, k, binarize)
    if edge == "hybrid":
        return prepare(hybrid_adjacency(spatial, functional), k, binarize)
    if edge == "pswe":
        return prepare(pswe, k, binarize)
    if edge == "random":
        sparse = knn_sparsify(hybrid_adjacency(spatial, functional), k)
        shuffled = degree_preserving_shuffle(sparse, seed)
        return normalize((shuffled > 0).astype(np.float32) if binarize else shuffled)
    raise ValueError(f"unknown edge type: {edge}")
