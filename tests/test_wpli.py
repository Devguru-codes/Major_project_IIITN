import numpy as np

from eegrep.connectivity import wpli

SF = 128.0
T = np.arange(1280) / SF


def _pairs(rng, lag, n=20, noise=0.3):
    """n windows of two 10 Hz channels with a fixed phase lag and independent noise."""
    a = np.sin(2 * np.pi * 10 * T) + noise * rng.standard_normal((n, T.size))
    b = np.sin(2 * np.pi * 10 * T - lag) + noise * rng.standard_normal((n, T.size))
    return np.stack([a, b], axis=1)


def test_consistent_phase_lag_gives_high_wpli(rng):
    w = wpli(_pairs(rng, lag=np.pi / 2), SF, (8.0, 13.0))
    assert w[:, 0, 1].mean() > 0.9


def test_zero_lag_volume_conduction_gives_low_wpli(rng):
    w = wpli(_pairs(rng, lag=0.0), SF, (8.0, 13.0))
    assert w[:, 0, 1].mean() < 0.3


def test_wpli_is_symmetric_bounded_zero_diagonal(rng):
    x = rng.standard_normal((4, 6, 1280))
    w = wpli(x, SF, (4.0, 8.0))
    assert np.allclose(w, w.transpose(0, 2, 1), atol=1e-5)
    assert w.min() >= 0 and w.max() <= 1 + 1e-6
    assert np.all(np.diagonal(w, axis1=1, axis2=2) == 0)
