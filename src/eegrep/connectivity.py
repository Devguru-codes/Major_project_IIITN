"""Band-limited weighted phase-lag index (wPLI) per window.

For band-passed analytic signals z_i(t), the imaginary cross-spectrum is
Im(z_i z_j*) = y_i x_j - x_i y_j  (z = x + iy). Within one window,
    wPLI_ij = |mean_t Im(z_i z_j*)| / mean_t |Im(z_i z_j*)|      (Vinck et al. 2011)
estimated over time samples. Zero-lag (volume-conducted) coupling has
Im = 0 and therefore wPLI ~ 0.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def wpli(windows: np.ndarray, sfreq: float, band: tuple[float, float], chunk: int = 32) -> np.ndarray:
    """windows: (N, C, T) -> (N, C, C) wPLI in [0, 1], zero diagonal."""
    z = hilbert(bandpass(windows, sfreq, *band), axis=-1)
    x, y = z.real.astype(np.float32), z.imag.astype(np.float32)
    n, c, _ = windows.shape
    out = np.empty((n, c, c), dtype=np.float32)
    for s in range(0, n, chunk):
        xs, ys = x[s:s + chunk], y[s:s + chunk]
        im = ys[:, :, None, :] * xs[:, None, :, :] - xs[:, :, None, :] * ys[:, None, :, :]
        num = np.abs(im.mean(-1))
        den = np.abs(im).mean(-1)
        out[s:s + chunk] = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    idx = np.arange(c)
    out[:, idx, idx] = 0.0
    return out


def wpli_bands(windows: np.ndarray, sfreq: float, bands: dict) -> np.ndarray:
    """(N, C, T) -> (N, B, C, C) for bands in dict order."""
    return np.stack([wpli(windows, sfreq, tuple(b)) for b in bands.values()], axis=1)
