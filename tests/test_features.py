import numpy as np
import pytest

from eegrep.features import N_FEATURES


@pytest.mark.parametrize("p", ["P1", "P2", "P3", "P4", "P5", "P6", "P7"])
def test_feature_shapes_and_finite(small_cache, p):
    x = small_cache.X(p)
    n = len(small_cache.subject)
    expected = 1280 if p == "P5" else N_FEATURES[p]
    assert x.shape == (n, 19, expected)
    assert np.isfinite(x).all()


def test_relative_band_power_sums_to_one(small_cache):
    rel = small_cache.X("P1")[..., :5].sum(-1)
    assert np.allclose(rel, 1.0, atol=1e-3)


def test_synthetic_slowing_is_visible(small_cache):
    """AD (label 0) has more theta / less alpha than CN (label 2) in the generator."""
    p1 = small_cache.X("P1")
    ratio = p1[..., 1].mean(-1) / p1[..., 2].mean(-1)          # theta / alpha
    assert ratio[small_cache.label == 0].mean() > ratio[small_cache.label == 2].mean()


def test_pswe_features_flag_ad_bursts(small_cache):
    frac = small_cache.X("P7")[..., 0].mean(-1)
    assert frac[small_cache.label == 0].mean() > frac[small_cache.label == 2].mean()
