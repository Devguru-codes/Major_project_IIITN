import numpy as np
import pytest

from eegrep import pswe

SF = 128.0


def _recording(rng, burst_s=0.0, onset=20.0, duration=60.0):
    t = np.arange(int(duration * SF)) / SF
    x = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.standard_normal(t.size)
    if burst_s:
        m = (t >= onset) & (t < onset + burst_s)
        x[m] = 3 * np.sin(2 * np.pi * 3 * t[m]) + 0.2 * rng.standard_normal(m.sum())
    return np.stack([x, x])


def test_runs_below_counts_consecutive_seconds():
    mpf = np.array([9, 5, 5, 5, 5, 5, 9, 5, 5, 5, 5, 9], dtype=float)
    assert pswe.runs_below(mpf, 6.0, 5) == [(1, 6)]           # 5-s run kept, 4-s run dropped


def test_long_slow_burst_is_detected(cfg, rng):
    events, mpf, mask, _ = pswe.detect(_recording(rng, burst_s=10.0), SF, cfg["pswe"])
    assert len(events) == 2                                     # one per channel
    assert np.all(np.abs(events.onset_s - 20.0) <= 2.0)
    assert np.all(events.duration_s >= 8.0) and np.all(events.mean_mpf < 6.0)


def test_short_slow_burst_is_not_detected(cfg, rng):
    events, *_ = pswe.detect(_recording(rng, burst_s=2.0), SF, cfg["pswe"])
    assert events.empty


def test_alpha_background_has_no_events_and_mpf_near_10hz(cfg, rng):
    events, mpf, *_ = pswe.detect(_recording(rng), SF, cfg["pswe"])
    assert events.empty
    assert 8.0 < np.median(mpf) < 12.0


def _slow_background(rng, burst=False, duration=60.0):
    """Continuous 5 Hz background (already below the fixed 6 Hz threshold); optional 10 s 2 Hz burst at 20 s."""
    t = np.arange(int(duration * SF)) / SF
    x = np.sin(2 * np.pi * 5 * t) + 0.2 * rng.standard_normal(t.size)
    if burst:
        m = (t >= 20) & (t < 30)
        x[m] = 3 * np.sin(2 * np.pi * 2 * t[m]) + 0.2 * rng.standard_normal(m.sum())
    return np.stack([x, x])


def test_fixed_threshold_flags_continuous_slowing_but_relative_does_not(cfg, rng):
    x = _slow_background(rng)
    events, mpf, *_ = pswe.detect(x, SF, cfg["pswe"])
    assert len(events) == 2 and events.duration_s.min() > 50           # whole recording = one "event"
    rel, _ = pswe.events_below(mpf, pswe.relative_thresholds(mpf, drop_hz=2.0), 1.0, 5.0, ["a", "b"])
    assert rel.empty


@pytest.mark.parametrize("spec", [{"drop_hz": 2.0}, {"mad_k": 3.0}])
def test_relative_detector_finds_transient_drop_on_slow_background(cfg, rng, spec):
    x = _slow_background(rng, burst=True)
    _, mpf, *_ = pswe.detect(x, SF, cfg["pswe"])
    rel, mask = pswe.events_below(mpf, pswe.relative_thresholds(mpf, **spec), 1.0, 5.0, ["a", "b"])
    assert len(rel) == 2 and np.all(np.abs(rel.onset_s - 20) <= 2) and np.all(rel.duration_s >= 8)


def test_window_features_shape_and_fraction(cfg, rng):
    x = _recording(rng, burst_s=10.0)
    events, mpf, mask, step = pswe.detect(x, SF, cfg["pswe"])
    starts = np.arange(6) * 10.0
    f = pswe.window_features(mpf, mask, events, ["a", "b"], starts, 10.0, step, 60.0, 60.0)
    assert f.shape == (6, 2, 4)
    assert f[2, 0, 0] > 0.5 and f[0, 0, 0] == 0.0             # window 20–30 s is mostly inside the event
