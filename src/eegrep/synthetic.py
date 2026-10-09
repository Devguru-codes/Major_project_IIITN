"""Synthetic resting-state EEG with class-dependent structure, for tests and the
NB00 smoke run only (never used in any reported result).

CN: strong 10 Hz alpha · FTD: weaker alpha, more theta · AD: weakest alpha, most
theta, plus 8 s bursts of 3 Hz activity (synthetic PSWE-like events). A shared
alpha source with channel-specific phase lags gives non-trivial wPLI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE = {  # label: (alpha amp, theta amp, slow bursts)
    0: (0.6, 1.5, True),    # AD
    1: (1.2, 1.0, False),   # FTD
    2: (2.0, 0.5, False),   # CN
}
GROUP = {0: "A", 1: "F", 2: "C"}


def pink_noise(rng, shape, sfreq):
    white = rng.standard_normal(shape)
    spec = np.fft.rfft(white, axis=-1)
    f = np.fft.rfftfreq(shape[-1], 1 / sfreq)
    spec /= np.sqrt(np.maximum(f, 0.5))
    return np.fft.irfft(spec, n=shape[-1], axis=-1)


def make_recordings(n_per_class: int = 7, duration_s: float = 60.0, sfreq: float = 128.0,
                    n_channels: int = 19, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sfreq)) / sfreq
    recordings, rows = [], []
    sid = 0
    for label in (0, 1, 2):
        alpha_amp, theta_amp, bursts = PROFILE[label]
        for _ in range(n_per_class):
            sid += 1
            subject = f"sub-{sid:03d}"
            lags = rng.uniform(0, np.pi / 2, n_channels)[:, None]
            x = pink_noise(rng, (n_channels, len(t)), sfreq)
            x += alpha_amp * np.sin(2 * np.pi * 10 * t + lags)
            x += theta_amp * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi, (n_channels, 1)))
            if bursts:
                for onset in rng.choice(np.arange(5, duration_s - 15, 10), size=2, replace=False):
                    m = (t >= onset) & (t < onset + 8)
                    x[: n_channels // 2, m] += 6.0 * np.sin(2 * np.pi * 3 * t[m])
            recordings.append((subject, x.astype(np.float32), sfreq))
            rows.append({"participant_id": subject, "Gender": rng.choice(["M", "F"]),
                         "Age": int(rng.integers(55, 80)), "Group": GROUP[label],
                         "MMSE": {0: 18, 1: 22, 2: 30}[label]})
    return recordings, pd.DataFrame(rows)
