import numpy as np
import pytest

from eegrep import graphs


@pytest.fixture(scope="module")
def coords(cfg):
    return graphs.electrode_positions(cfg["dataset"]["channels"], cfg["dataset"]["channel_rename"])


def test_spatial_adjacency(coords):
    s = graphs.spatial_adjacency(coords)
    assert s.shape == (19, 19)
    assert np.allclose(s, s.T) and np.all(np.diag(s) == 0)
    assert s.max() <= 1 and s.min() >= 0


@pytest.mark.parametrize("k", [3, 4, 6])
def test_knn_symmetric_with_min_degree(rng, k):
    a = rng.random((5, 19, 19)).astype(np.float32)
    a = (a + a.transpose(0, 2, 1)) / 2
    s = graphs.knn_sparsify(a, k)
    assert np.allclose(s, s.transpose(0, 2, 1))
    assert np.all((s > 0).sum(-1) >= k)
    assert np.all(np.diagonal(s, axis1=1, axis2=2) == 0)


def test_normalized_spectral_radius_at_most_one(rng):
    a = graphs.knn_sparsify(rng.random((19, 19)).astype(np.float32), 4)
    eig = np.linalg.eigvalsh(graphs.normalize(a))
    assert np.abs(eig).max() <= 1 + 1e-5


def test_build_all_edge_types(coords, rng):
    n = 4
    func = rng.random((n, 19, 19)).astype(np.float32)
    func = (func + func.transpose(0, 2, 1)) / 2
    spatial = graphs.spatial_adjacency(coords)
    for edge in ["spatial", "functional", "hybrid", "pswe", "identity", "random"]:
        a = graphs.build(edge, spatial=spatial, functional=func, pswe=func, k=4)
        assert a.shape in [(19, 19), (n, 19, 19)]
        assert np.allclose(a, np.swapaxes(a, -1, -2), atol=1e-6)
    assert np.allclose(graphs.build("identity", spatial=spatial), np.eye(19))


def test_random_graph_preserves_degree_sequence(rng):
    a = graphs.knn_sparsify(rng.random((19, 19)).astype(np.float32), 4)
    r = graphs.degree_preserving_shuffle(a, seed=3)
    assert sorted((a > 0).sum(1)) == sorted((r > 0).sum(1))
    assert not np.array_equal(a, r)


def test_jaccard_bounds():
    m = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    j = graphs.jaccard_adjacency(m)
    assert j[0, 1] == pytest.approx(0.5) and j[0, 2] == 0 and np.all(np.diag(j) == 0)
