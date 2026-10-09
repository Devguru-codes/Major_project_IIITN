import numpy as np
import pytest

from eegrep import graphs, synthetic
from eegrep.cache import PIPELINES, FeatureCache, build_cache, merge_caches


def test_shard_caches_merge_into_one(cfg, tmp_path):
    recs, part = synthetic.make_recordings(n_per_class=2, duration_s=30.0, seed=2)
    labels = dict(zip(part["participant_id"], part["Group"].map(cfg["dataset"]["group_to_label"])))
    coords = graphs.electrode_positions(cfg["dataset"]["channels"], cfg["dataset"]["channel_rename"])
    a = build_cache(recs[::2], labels, cfg, coords, tmp_path / "a")
    b = build_cache(recs[1::2], labels, cfg, coords, tmp_path / "b")
    merged = FeatureCache(merge_caches([a, b], tmp_path / "m"))
    ca, cb = FeatureCache(a), FeatureCache(b)
    assert len(merged.subject) == len(ca.subject) + len(cb.subject)
    for p in PIPELINES:
        assert merged.X(p).shape[0] == len(merged.subject)
    assert merged.pswe_per_window().shape == (len(merged.subject), 19, 19)
    with pytest.raises(AssertionError):
        merge_caches([a, a], tmp_path / "dup")


def test_windowize_non_overlapping():
    from eegrep.cache import windowize

    x = np.arange(2 * 3000, dtype=np.float32).reshape(2, 3000)
    w, starts = windowize(x, 128.0, 10.0)
    assert w.shape == (2, 2, 1280) and list(starts) == [0.0, 10.0]
    assert w[1, 0, 0] == x[0, 1280]
