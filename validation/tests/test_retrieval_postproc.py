"""The Tier-1 techniques. RESULTS.md's negative result is only meaningful if these are correct."""
import numpy as np
import pytest

import retrieval_tcga_ot_kreciprocal as kr
import retrieval_tcga_ot_query_expansion as qe
import retrieval_tcga_ot_whitening as wh
from retrieval_common import baseline_ranking, cosine_sim, eval_ranking, l2_normalize


def _data(seed=0, n_db=80, n_q=30, dim=12, n_cls=4):
    rng = np.random.default_rng(seed)
    db_l = np.array([f"C{i % n_cls}" for i in range(n_db)])
    q_l = np.array([f"C{i % n_cls}" for i in range(n_q)])
    ctr = rng.normal(size=(n_cls, dim)) * 2
    Xdb = np.stack([ctr[int(l[1:])] + rng.normal(size=dim) for l in db_l]).astype(np.float32)
    Xq = np.stack([ctr[int(l[1:])] + rng.normal(size=dim) for l in q_l]).astype(np.float32)
    return Xdb, db_l, Xq, q_l


# ---------------------------------------------------------------- whitening

def test_full_rank_pca_is_a_pure_rotation_so_it_cannot_change_the_ranking():
    """pca-<dim> (no truncation, no rescaling) is an orthonormal rotation of the centered data.
    Rotations preserve inner products and norms, so cosine ranking must be IDENTICAL to plain
    centering. If this fails, fit_pca's eigenvectors are not orthonormal / not column-major."""
    Xdb, db_l, Xq, q_l = _data()
    mu, lam, V = wh.fit_pca(Xdb)
    m_center = wh.score(wh.make_transform("center", None, mu, lam, V), Xdb, db_l, Xq, q_l)
    m_pca = wh.score(wh.make_transform("pca", Xdb.shape[1], mu, lam, V), Xdb, db_l, Xq, q_l)
    assert m_center == m_pca


def test_pca_eigenvalues_are_sorted_descending():
    _, lam, _ = wh.fit_pca(_data()[0])
    assert np.all(np.diff(lam) <= 1e-9)


def test_pcaw_actually_whitens_the_database():
    """After full-rank PCA-whitening the database covariance must be ~identity. This is the
    property the whole 'whitening hurts' conclusion is predicated on."""
    Xdb = _data(n_db=400, dim=8)[0]
    mu, lam, V = wh.fit_pca(Xdb)
    Z = wh.make_transform("pcaw", 8, mu, lam, V)(Xdb)
    cov = np.cov(Z, rowvar=False)
    assert np.allclose(cov, np.eye(8), atol=0.05)


def test_zca_whitens_and_stays_in_the_original_axes():
    Xdb = _data(n_db=400, dim=8)[0]
    mu, lam, V = wh.fit_pca(Xdb)
    Z = wh.make_transform("zca", None, mu, lam, V)(Xdb)
    assert Z.shape == Xdb.shape                      # ZCA does not reduce dimension
    assert np.allclose(np.cov(Z, rowvar=False), np.eye(8), atol=0.05)


def test_transform_is_fit_on_the_database_only():
    """mu/lam/V must be a function of Xdb alone -- a query must never influence the transform,
    or the val/test discipline is void."""
    Xdb, _, Xq, _ = _data()
    a = wh.fit_pca(Xdb)
    b = wh.fit_pca(Xdb)                              # same input -> same fit
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    c = wh.fit_pca(np.vstack([Xdb, Xq]))             # different input -> different fit
    assert not np.allclose(a[0], c[0])


# ---------------------------------------------------------- k-reciprocal

def test_lambda_one_reproduces_raw_cosine_exactly():
    """THE sanity check the script only prints. λ=1 means 'ignore the Jaccard term', so the
    final ranking must be the plain cosine ranking, bit for bit."""
    Xdb, db_l, Xq, q_l = _data()
    k1, k2 = 5, 3
    original_dist, initial_rank, qn = kr.precompute(Xq, Xdb, top_m=k1 + 1)
    V = kr.build_V(original_dist, initial_rank, k1)
    jac = kr.jaccard_distance(kr.smooth_V(V, initial_rank, k2), qn)

    final = jac * (1 - 1.0) + original_dist[:qn] * 1.0
    got = eval_ranking(-final[:, qn:], db_l, q_l)
    want = baseline_ranking(Xdb, db_l, Xq, q_l)
    assert got == want


def test_k_reciprocal_neighbours_are_mutual():
    """b is a k-reciprocal neighbour of a only if each is in the other's top-k. A one-way
    implementation would silently pass everything else in this file."""
    Xdb, _, Xq, _ = _data()
    original_dist, initial_rank, _ = kr.precompute(Xq, Xdb, top_m=11)
    k = 5
    for i in (0, 3, 17):
        for j in kr.k_reciprocal_neighbors(initial_rank, i, k):
            assert i in initial_rank[j, :k + 1], f"{j} is not reciprocal with {i}"


def test_jaccard_distance_is_in_the_unit_interval():
    Xdb, _, Xq, _ = _data()
    original_dist, initial_rank, qn = kr.precompute(Xq, Xdb, top_m=11)
    jac = kr.jaccard_distance(kr.smooth_V(kr.build_V(original_dist, initial_rank, 10),
                                          initial_rank, 3), qn)
    assert np.isfinite(jac).all()
    assert jac.min() >= -1e-6 and jac.max() <= 1.0 + 1e-6


# ------------------------------------------------------------ QE / DBA

def test_dba_without_self_exclusion_is_a_no_op():
    """Documented in the script's own docstring: 'a database vector is its own nearest
    neighbour, so DBA's neighbour search must skip the diagonal or it degenerates toward the
    identity'. Prove both halves."""
    Xdb_n = l2_normalize(_data()[0])
    no_excl = qe._expand(Xdb_n, Xdb_n, k=1, alpha=1.0, exclude_self=False)
    assert np.allclose(no_excl, Xdb_n, atol=1e-5)        # x + 1.0*x, renormalized == x

    with_excl = qe.apply_dba(Xdb_n, k=1, alpha=1.0)
    assert not np.allclose(with_excl, Xdb_n, atol=1e-3)  # the real DBA moves the vectors


def test_alpha_zero_is_an_unweighted_mean():
    """alpha=0 -> every neighbour weight is 1 -> plain mean of self + top-k."""
    Xdb_n = l2_normalize(_data()[0])
    Xq_n = l2_normalize(_data()[2])
    k = 3
    got = qe.apply_qe(Xdb_n, Xq_n, k=k, alpha=0.0)

    sim = Xq_n @ Xdb_n.T
    nn = np.argsort(-sim, axis=1)[:, :k]
    want = l2_normalize(Xq_n + Xdb_n[nn].sum(axis=1))
    assert np.allclose(got, want, atol=1e-5)


def test_qe_output_rows_are_unit_norm():
    Xdb_n = l2_normalize(_data()[0])
    Xq_n = l2_normalize(_data()[2])
    Z = qe.apply_qe(Xdb_n, Xq_n, k=5, alpha=2.0)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)


def test_negative_similarities_are_clamped_not_sign_flipped():
    """The docstring promises: 'Similarities are clamped at 0 so a negative cosine can never flip
    a neighbour's sign under an even alpha.'"""
    X = l2_normalize(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    Z = qe._expand(X, X, k=2, alpha=2.0, exclude_self=True)
    assert np.isfinite(Z).all()
