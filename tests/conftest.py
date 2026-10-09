import numpy as np
import pytest

from eegrep import graphs, synthetic
from eegrep.cache import FeatureCache, build_cache
from eegrep.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_cache(cfg, tmp_path_factory):
    """3 subjects per class, 40 s each -> 36 windows. Built with the real cache code path."""
    recs, part = synthetic.make_recordings(n_per_class=3, duration_s=40.0, seed=1)
    labels = dict(zip(part["participant_id"], part["Group"].map(cfg["dataset"]["group_to_label"])))
    coords = graphs.electrode_positions(cfg["dataset"]["channels"], cfg["dataset"]["channel_rename"])
    out = build_cache(recs, labels, cfg, coords, tmp_path_factory.mktemp("cache"))
    return FeatureCache(out)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
