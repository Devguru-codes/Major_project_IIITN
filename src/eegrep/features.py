"""Factor A: node-feature representations. Every function maps windows
(N, C, T) at `sfreq` (plus precomputed wPLI where needed) to X ∈ (N, C, F).

P1 band power (7) · P2 connectivity strength (5) · P3 DWT (20) · P4 time-domain (10)
P5 raw z-scored (T) · P6 fused (34) · P7 PSWE (4, built in pswe.window_features)
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import welch
from scipy.stats import kurtosis, skew

N_FEATURES = {"P1": 7, "P2": 5, "P3": 20, "P4": 10, "P6": 34, "P7": 4}  # P5 = window length in samples


def p1_band(windows: np.ndarray, sfreq: float, bands: dict, nperseg_s: float, peak_range) -> np.ndarray:
    """Relative power in 5 bands, normalised spectral entropy, peak frequency in `peak_range`."""
    nper = int(nperseg_s * sfreq)
    f, psd = welch(windows, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)
    lo, hi = list(bands.values())[0][0], list(bands.values())[-1][1]
    full = (f >= lo) & (f < hi)
    total = psd[..., full].sum(-1)
    rel = [psd[..., (f >= b[0]) & (f < b[1])].sum(-1) / total for b in bands.values()]
    p = psd[..., full] / total[..., None]
    entropy = -(p * np.log(p + 1e-12)).sum(-1) / np.log(full.sum())
    pk = (f >= peak_range[0]) & (f <= peak_range[1])
    peak = f[pk][np.argmax(psd[..., pk], axis=-1)]
    return np.stack(rel + [entropy, peak], axis=-1).astype(np.float32)


def p2_conn(wpli_b: np.ndarray) -> np.ndarray:
    """wPLI node strength per band: (N, B, C, C) -> (N, C, B)."""
    c = wpli_b.shape[-1]
    return (wpli_b.sum(-1) / (c - 1)).transpose(0, 2, 1).astype(np.float32)


def p3_dwt(windows: np.ndarray, wavelet: str, level: int, keep: list[str]) -> np.ndarray:
    """Per kept sub-band: relative energy, normalised coefficient entropy, mean |c|, std."""
    coeffs = pywt.wavedec(windows, wavelet, level=level, axis=-1)
    names = [f"A{level}"] + [f"D{level - i}" for i in range(level)]       # A5, D5, D4, ..., D1
    bands = [coeffs[names.index(k)] for k in keep]
    energies = [(c ** 2).mean(-1) for c in bands]
    total = np.sum(energies, axis=0) + 1e-12
    feats = []
    for c, e in zip(bands, energies):
        p = c ** 2 / ((c ** 2).sum(-1, keepdims=True) + 1e-12)
        ent = -(p * np.log(p + 1e-12)).sum(-1) / np.log(c.shape[-1])
        feats += [e / total, ent, np.abs(c).mean(-1), c.std(-1)]
    return np.stack(feats, axis=-1).astype(np.float32)


def p4_time(windows: np.ndarray, sampen_order: int, sampen_r: float) -> np.ndarray:
    """mean, var, RMS, skew, kurtosis, ZCR, Hjorth mobility, Hjorth complexity,
    sample entropy, line length. (Hjorth activity == var, so line length replaces it.)"""
    import antropy

    x = windows
    dx, ddx = np.diff(x, axis=-1), np.diff(x, n=2, axis=-1)
    var = x.var(-1)
    mob = np.sqrt(dx.var(-1) / (var + 1e-12))
    comp = np.sqrt(ddx.var(-1) / (dx.var(-1) + 1e-12)) / (mob + 1e-12)
    zcr = (np.diff(np.signbit(x - x.mean(-1, keepdims=True)), axis=-1)).mean(-1)
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    sampen = np.array([antropy.sample_entropy(s, order=sampen_order, tolerance=sampen_r * s.std())
                       for s in flat]).reshape(x.shape[:-1])
    sampen = np.nan_to_num(sampen, nan=0.0, posinf=0.0)
    feats = [x.mean(-1), var, np.sqrt((x ** 2).mean(-1)), skew(x, axis=-1), kurtosis(x, axis=-1),
             zcr, mob, comp, sampen, np.abs(dx).mean(-1)]
    return np.stack(feats, axis=-1).astype(np.float32)


def p5_raw(windows: np.ndarray) -> np.ndarray:
    mu = windows.mean(-1, keepdims=True)
    sd = windows.std(-1, keepdims=True) + 1e-8
    return ((windows - mu) / sd).astype(np.float32)


def clustering_weighted(w: np.ndarray) -> np.ndarray:
    """Onnela weighted clustering on a complete weighted graph: (N, C, C) -> (N, C)."""
    w = w / (w.max(axis=(-1, -2), keepdims=True) + 1e-12)
    c3 = np.cbrt(w)
    tri = np.einsum("nij,njk,nki->ni", c3, c3, c3)
    k = (w > 0).sum(-1)
    return (tri / np.maximum(k * (k - 1), 1)).astype(np.float32)


def eigenvector_centrality(w: np.ndarray) -> np.ndarray:
    _, vecs = np.linalg.eigh(w.astype(np.float64))
    v = np.abs(vecs[..., -1])
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)).astype(np.float32)


def p6_fused(p1: np.ndarray, p3: np.ndarray, p2: np.ndarray, wpli_alpha: np.ndarray) -> np.ndarray:
    extra = np.stack([clustering_weighted(wpli_alpha), eigenvector_centrality(wpli_alpha)], axis=-1)
    return np.concatenate([p1, p3, p2, extra], axis=-1).astype(np.float32)


def compute_all(windows: np.ndarray, sfreq: float, wpli_b: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """P1–P6 for a batch of windows (P7 comes from the PSWE detector on the continuous recording)."""
    fc, bands = cfg["features"], cfg["bands"]
    alpha = wpli_b[:, list(bands).index(cfg["graph"]["functional_band"])]
    p1 = p1_band(windows, sfreq, bands, fc["welch_nperseg_s"], fc["peak_freq_range"])
    p2 = p2_conn(wpli_b)
    p3 = p3_dwt(windows, fc["dwt_wavelet"], fc["dwt_level"], fc["dwt_keep"])
    return {"P1": p1, "P2": p2, "P3": p3,
            "P4": p4_time(windows, fc["sampen_order"], fc["sampen_r"]),
            "P5": p5_raw(windows),
            "P6": p6_fused(p1, p3, p2, alpha)}
