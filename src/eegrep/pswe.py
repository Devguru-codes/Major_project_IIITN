"""Paroxysmal slow-wave event (PSWE) detection: a BBB-dysfunction-associated EEG marker.

Definition (Milikovsky et al. 2019, 10.1126/scitranslmed.aaw8954; parameters as
reported in Milikovsky et al. 2023, 10.3390/s23020918): median power frequency
(MPF) from FFTs of 2 s windows with 1 s overlap; an event is MPF < 6 Hz for five
consecutive seconds or more.

Here the MPF series is sampled once per step (1 s); each sample stands for one
second, so a run of n consecutive sub-threshold samples is an event of n·step
seconds. Detection is per recording and label-free (no cross-subject leakage).
ds004504 has no BBB imaging: PSWEs are an *associated marker*, not a BBB measure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def median_power_frequency(x: np.ndarray, sfreq: float, win_s: float, step_s: float,
                           fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """x: (C, T) -> mpf (C, S), segment start times (S,)."""
    win, step = int(round(win_s * sfreq)), int(round(step_s * sfreq))
    n_seg = 1 + (x.shape[-1] - win) // step
    starts = np.arange(n_seg) * step
    segs = np.stack([x[:, s:s + win] for s in starts], axis=1)          # (C, S, win)
    spec = np.abs(np.fft.rfft(segs * np.hanning(win), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(win, 1.0 / sfreq)
    band = (freqs >= fmin) & (freqs <= fmax)
    cum = np.cumsum(spec[..., band], axis=-1)
    half = cum[..., -1:] / 2.0
    mpf = freqs[band][np.argmax(cum >= half, axis=-1)]
    return mpf.astype(np.float32), starts / sfreq


def runs_below(mpf_1d: np.ndarray, threshold: float, min_len: int) -> list[tuple[int, int]]:
    """[start, stop) index runs where mpf < threshold for at least min_len samples."""
    below = np.concatenate([[False], mpf_1d < threshold, [False]])
    edges = np.flatnonzero(np.diff(below.astype(np.int8)))
    return [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_len]


def detect(x: np.ndarray, sfreq: float, cfg: dict, channels: list[str] | None = None):
    """x: (C, T) continuous preprocessed recording.

    Returns (events DataFrame, mpf (C, S), in_event mask (C, S), step_s).
    """
    mpf, starts = median_power_frequency(x, sfreq, cfg["win_s"], cfg["step_s"], cfg["fmin"], cfg["fmax"])
    min_len = int(np.ceil(cfg["min_duration_s"] / cfg["step_s"]))
    channels = channels or [f"ch{i}" for i in range(x.shape[0])]
    mask = np.zeros(mpf.shape, dtype=bool)
    rows = []
    for c, ch in enumerate(channels):
        for s, e in runs_below(mpf[c], cfg["mpf_threshold_hz"], min_len):
            mask[c, s:e] = True
            rows.append({"channel": ch, "onset_s": float(starts[s]), "duration_s": (e - s) * cfg["step_s"],
                         "mean_mpf": float(mpf[c, s:e].mean()), "min_mpf": float(mpf[c, s:e].min())})
    events = pd.DataFrame(rows, columns=["channel", "onset_s", "duration_s", "mean_mpf", "min_mpf"])
    return events, mpf, mask, cfg["step_s"]


def subject_summary(events: pd.DataFrame, mask: np.ndarray, channels: list[str], duration_s: float) -> pd.DataFrame:
    """Per-channel PSWE rate (events/min) and fraction of time in PSWE."""
    counts = events.groupby("channel").size().reindex(channels, fill_value=0)
    return pd.DataFrame({"channel": channels,
                         "rate_per_min": counts.values / (duration_s / 60.0),
                         "time_fraction": mask.mean(axis=1)})


def window_features(mpf: np.ndarray, mask: np.ndarray, events: pd.DataFrame, channels: list[str],
                    window_starts_s: np.ndarray, window_s: float, step_s: float,
                    local_rate_s: float, duration_s: float) -> np.ndarray:
    """P7 node features per window: (N, C, 4) =
    [fraction of window in PSWE, mean MPF, min MPF, local PSWE onset rate (events/min, ±local_rate_s)].
    """
    n, c = len(window_starts_s), mpf.shape[0]
    out = np.zeros((n, c, 4), dtype=np.float32)
    onsets = {ch: events.loc[events.channel == ch, "onset_s"].to_numpy() for ch in channels}
    for i, t0 in enumerate(window_starts_s):
        s0 = int(t0 / step_s)
        s1 = max(s0 + 1, min(mpf.shape[1], int((t0 + window_s) / step_s)))
        lo, hi = max(0.0, t0 - local_rate_s), min(duration_s, t0 + window_s + local_rate_s)
        span_min = (hi - lo) / 60.0
        out[i, :, 0] = mask[:, s0:s1].mean(1)
        out[i, :, 1] = mpf[:, s0:s1].mean(1)
        out[i, :, 2] = mpf[:, s0:s1].min(1)
        for j, ch in enumerate(channels):
            o = onsets[ch]
            out[i, j, 3] = np.count_nonzero((o >= lo) & (o < hi)) / span_min
    return out
