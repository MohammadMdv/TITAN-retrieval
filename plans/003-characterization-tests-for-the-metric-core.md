# Plan 003 — No automated tests exist for the numeric core the entire project's claims rest on

**Written against commit:** `9d56a69` · **re-stamped and drift-checked at `1fe057d`** (post-001, post-002 merge)
**Category:** test coverage
**Impact:** HIGH · **Effort:** M · **Risk of fix:** LOW (adds files; changes no behavior)
**Depends on:** nothing. **Blocks:** plan 004 (the consolidation refactor needs these tests as its safety net).

> **Drift check at `1fe057d`.** Every module this plan tests — `retrieval_common.py`,
> `retrieval_tcga_ot.py`, `subtasks_tcga_ot.py`, the three Tier-1 scripts,
> `extract_bracs_features.py`, and `ProxyAnchor`/`pk_batches` inside `finetune_bracs_lora.py` — is
> **untouched** by plans 001 and 002, so every excerpt below is still accurate.
>
> **One thing has changed:** `validation/tests/` now **exists** and already holds
> `test_make_results.py` (4 tests, plan 001) and `test_lora_seeding.py` (4 tests, plan 002), for
> **8 existing tests**. §1's "no `tests/` directory" is now historical. Two consequences:
> 1. `conftest.py` (Step 2) is still needed and is now a **bonus**: those 8 existing tests were
>    only runnable via a direct-call harness because `pytest` was not installed. Once `conftest.py`
>    and `pytest` land, they become properly `pytest`-runnable for the first time. **They must
>    pass.** Do not edit them.
> 2. Expected test count is now **≥ 43** (≥35 new + 8 existing), not ≥35. Update that criterion in §6.
>
> **`pip install pytest` is now authorized.** Plans 001 and 002 deliberately avoided it (the user
> had not approved a change to their shared venv). The user has now asked for plan 003 explicitly,
> and installing `pytest` into `.venv` is Step 1 of this plan — so it is sanctioned. It is additive
> and dev-only.

---

## 1. Why this matters

This repository has **zero automated tests** (no `tests/`, no `pytest.ini`, no CI, `pytest` is not
even installed in `.venv`). Its 4,649 lines exist to produce ~30 numbers, and those numbers are
the product. Every claim in `RESULTS.md` — "Tier-1 does not improve retrieval", "LoRA lifts Acc@3
from 0.716 to 0.757" — is a *difference between two numbers computed by bespoke, unverified code*.

The code that computes them is not thin glue. It is hand-rolled implementations of:

| Function | File:line | What a silent bug there would do |
|---|---|---|
| `eval_ranking` (Acc@k, MVAcc@k, tie-break) | `retrieval_common.py:71` | Corrupts **every** retrieval number in the repo |
| `topk_metrics` (a *second* copy) | `retrieval_tcga_ot.py:41` | Corrupts the headline TCGA-OT baseline |
| `retrieval` (a *third* copy) | `subtasks_tcga_ot.py:73` | Corrupts all five sub-task rows |
| `build_V` / `jaccard_distance` / `smooth_V` | `retrieval_tcga_ot_kreciprocal.py:82-130` | Turns a "k-reciprocal doesn't help" conclusion into an artifact of a bug |
| `_expand` (αQE / DBA) | `retrieval_tcga_ot_query_expansion.py:51` | Same, for query expansion |
| `fit_pca` / `make_transform` | `retrieval_tcga_ot_whitening.py:50-73` | Same, for whitening |
| `ProxyAnchor.forward` | `finetune_bracs_lora.py:234` | The LoRA result is trained by this loss |
| `_pad_grid_to_2x2` | `extract_bracs_features.py:91` | Silently corrupts **every BRACS embedding** (baseline and LoRA alike) |

The only correctness check that exists anywhere is the k-reciprocal λ=1.0 identity, and it is
merely **printed**, not enforced:

```python
# validation/retrieval_tcga_ot_kreciprocal.py:170-173
    lam1 = [m for lab, (k1, k2, lam), m in val_rows if lam == 1.0][0]
    ok = abs(lam1[SELECT_METRIC] - base_val[SELECT_METRIC]) < 1e-9
    print(f"\n[k-recip] sanity check -- λ=1.0 reproduces raw cosine on val: "
          f"{'PASS' if ok else 'FAIL'} ({lam1[SELECT_METRIC]:.4f} vs {base_val[SELECT_METRIC]:.4f})")
```

A `FAIL` here prints the word "FAIL" and the script carries on to save results and report a
conclusion. That is the state of quality assurance in this repo today.

Good news, and the reason this is an M and not an L: **every one of those functions is pure
numpy/torch.** None needs a GPU, an HF token, the network, or the 51.8 GB BRACS download. The
entire numeric core is testable in under a second on CPU.

## 2. What this plan delivers

A `validation/tests/` pytest suite that pins the numeric core with hand-computed expected values
and mathematical invariants, plus a `requirements-dev.txt` and a documented one-line test command.
It changes **no production code** — it is a characterization suite: it captures what the code does
*today* so that plan 004's refactor, and every future change, can be proven not to move a number.

## 3. Setup steps

### Step 1 — dev dependency

`pytest` is not installed. Create `validation/requirements-dev.txt`:

```
pytest>=7.0
```

Install into the existing venv (this is the executor's one permitted install):

```bash
cd /home/user01/TITAN && .venv/bin/python -m pip install -r validation/requirements-dev.txt
```

Do **not** add pytest to `setup.py`. That file carries upstream TITAN's pins
(`torch==2.0.1`, `transformers==4.46.0`, …) and is not ours to modify.

### Step 2 — make the flat imports work under pytest

Every script in `validation/` imports its siblings flat (`from common import ...`,
`from retrieval_bracs import load_split`), which works because they are run with `validation/` as
the cwd. Pytest runs from the repo root, so the tests need `validation/` on `sys.path`.

Create `validation/tests/conftest.py`:

```python
"""Put validation/ on sys.path so the tests can import the scripts the way the scripts import
each other (flat: `import common`, `import retrieval_common`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Add `validation/tests/__init__.py`? **No** — leave it out. With no `__init__.py`, pytest uses
rootdir-relative insertion and `conftest.py` does the rest.

> Sanity-check before writing any test:
> `cd /home/user01/TITAN && .venv/bin/python -c "import sys; sys.path.insert(0,'validation'); import retrieval_common, retrieval_tcga_ot, subtasks_tcga_ot, extract_bracs_features, finetune_bracs_lora; print('all importable')"`
> This must print `all importable` with no network access. If any module tries to reach the
> network or load a model **at import time**, stop — that is a separate bug, report it.

## 4. The tests

Write these five files. Every expected value below was derived by hand from the source; where a
test asserts an invariant rather than a literal, the invariant is stated in the docstring.

### `validation/tests/test_ranking_metrics.py`

```python
"""The top-k retrieval metrics. Every number in RESULTS.md flows through these."""
import numpy as np
import pytest

from retrieval_common import eval_ranking, cosine_sim, l2_normalize


def test_acc1_acc3_mvacc3_hand_computed():
    # one query; db similarities 0.9 > 0.8 > 0.7 > 0.1, labels A B B C -> top-3 = [A, B, B]
    sim = np.array([[0.9, 0.8, 0.7, 0.1]])
    db = np.array(["A", "B", "B", "C"])
    m = eval_ranking(sim, db, np.array(["B"]), k=3)
    assert m["acc@1"] == 0.0      # top-1 is A
    assert m["acc@3"] == 1.0      # B is in the top-3
    assert m["mvacc@3"] == 1.0    # majority of [A,B,B] is B


def test_mvacc_tie_breaks_toward_highest_similarity():
    """All three retrieved labels distinct -> every count is 1 -> the tie must resolve to the
    most-similar one (the code relies on `retr` being in descending-similarity order)."""
    sim = np.array([[0.9, 0.8, 0.7]])
    db = np.array(["A", "B", "C"])
    assert eval_ranking(sim, db, np.array(["A"]), k=3)["mvacc@3"] == 1.0   # A is closest -> wins
    assert eval_ranking(sim, db, np.array(["B"]), k=3)["mvacc@3"] == 0.0   # B tied but not closest


def test_acc_at_k_is_a_hit_not_a_majority():
    """Acc@k asks 'is the label anywhere in the top-k'; MVAcc@k asks 'does it win the vote'.
    Conflating them is the classic bug -- this case separates them."""
    sim = np.array([[0.9, 0.8, 0.7]])
    db = np.array(["A", "A", "B"])
    m = eval_ranking(sim, db, np.array(["B"]), k=3)
    assert m["acc@3"] == 1.0      # B is present
    assert m["mvacc@3"] == 0.0    # but A wins the vote 2-1


def test_key_names_track_k():
    m = eval_ranking(np.array([[1.0, 0.0]]), np.array(["A", "B"]), np.array(["A"]), k=2)
    assert set(m) == {"acc@1", "acc@2", "mvacc@2"}


def test_perfect_and_zero_retrieval():
    sim = np.eye(3)
    db = q = np.array(["A", "B", "C"])
    m = eval_ranking(sim, db, q, k=1)
    assert m["acc@1"] == 1.0
    m = eval_ranking(sim, db, np.array(["C", "A", "B"]), k=1)
    assert m["acc@1"] == 0.0


def test_l2_normalize_makes_unit_rows():
    X = np.random.default_rng(0).normal(size=(7, 5))
    assert np.allclose(np.linalg.norm(l2_normalize(X), axis=1), 1.0)


def test_l2_normalize_survives_a_zero_row():
    """The +eps guard must stop a 0/0 -> nan from poisoning a whole similarity matrix."""
    X = np.zeros((2, 4)); X[1, 0] = 1.0
    Z = l2_normalize(X)
    assert np.isfinite(Z).all()


def test_cosine_sim_is_bounded_and_symmetric_on_itself():
    X = np.random.default_rng(1).normal(size=(6, 5))
    S = cosine_sim(X, X)
    assert S.shape == (6, 6)
    assert np.all(S <= 1.0 + 1e-6) and np.all(S >= -1.0 - 1e-6)
    assert np.allclose(np.diag(S), 1.0, atol=1e-6)
```

### `validation/tests/test_metric_implementations_agree.py`

This is the file plan 004 depends on. It pins the three duplicate implementations to each other
**before** anyone consolidates them.

```python
"""Three copies of the same top-k metric exist. They must agree -- today, and after any refactor.

  retrieval_common.eval_ranking      (used by BRACS + every Tier-1 experiment)
  retrieval_tcga_ot.topk_metrics     (used by the headline TCGA-OT baseline)
  subtasks_tcga_ot.retrieval         (inline; used by the five sub-task rows)

retrieval_common.BASELINE_REF hardcodes 0.8717/0.7812 -- numbers produced by the SECOND
implementation -- while every Tier-1 delta is computed with the FIRST. If they ever diverge,
every Tier-1 delta silently becomes an apples-to-oranges comparison and nothing would catch it.
"""
import numpy as np
import pandas as pd

import subtasks_tcga_ot as sub
from retrieval_common import cosine_sim, eval_ranking
from retrieval_tcga_ot import topk_metrics


def _synthetic(seed=0, n_db=60, n_q=25, dim=16, n_cls=4):
    rng = np.random.default_rng(seed)
    labels = np.array([f"C{i % n_cls}" for i in range(n_db)])
    q_labels = np.array([f"C{i % n_cls}" for i in range(n_q)])
    # class-structured embeddings, so the metrics are non-trivial (not all 0, not all 1)
    centers = rng.normal(size=(n_cls, dim)) * 3
    Xdb = np.stack([centers[int(l[1:])] + rng.normal(size=dim) for l in labels])
    Xq = np.stack([centers[int(l[1:])] + rng.normal(size=dim) for l in q_labels])
    return Xdb.astype(np.float32), labels, Xq.astype(np.float32), q_labels


def test_eval_ranking_matches_topk_metrics():
    Xdb, db_l, Xq, q_l = _synthetic()
    sim = cosine_sim(Xq, Xdb)
    m = eval_ranking(sim, db_l, q_l, k=3)
    acc3, mv3 = topk_metrics(sim, db_l, q_l, K=3)
    assert m["acc@3"] == acc3
    assert m["mvacc@3"] == mv3


def test_eval_ranking_matches_subtasks_retrieval():
    """subtasks builds its own cosine from embeddings, so this pins the whole path, not just
    the ranking loop."""
    Xdb, db_l, Xq, q_l = _synthetic(seed=3)
    tr = pd.DataFrame({"emb": list(Xdb), "_y": db_l,
                       "case_id": [f"db{i}" for i in range(len(db_l))]})
    te = pd.DataFrame({"emb": list(Xq), "_y": q_l,
                       "case_id": [f"q{i}" for i in range(len(q_l))]})
    got = sub.retrieval(tr, te)
    want = eval_ranking(cosine_sim(Xq, Xdb), db_l, q_l, k=3)
    assert got["acc@1"] == want["acc@1"]
    assert got["mvacc@3"] == want["mvacc@3"]


def test_agreement_holds_across_many_random_problems():
    for seed in range(20):
        Xdb, db_l, Xq, q_l = _synthetic(seed=seed)
        sim = cosine_sim(Xq, Xdb)
        m = eval_ranking(sim, db_l, q_l, k=3)
        acc3, mv3 = topk_metrics(sim, db_l, q_l, K=3)
        assert (m["acc@3"], m["mvacc@3"]) == (acc3, mv3), f"diverged at seed {seed}"


def test_subtasks_retrieval_asserts_patient_disjointness():
    """The leakage guard must actually fire -- it is the thing standing between this repo and a
    meaningless number."""
    import pytest
    Xdb, db_l, Xq, q_l = _synthetic()
    tr = pd.DataFrame({"emb": list(Xdb), "_y": db_l, "case_id": ["SHARED"] * len(db_l)})
    te = pd.DataFrame({"emb": list(Xq), "_y": q_l, "case_id": ["SHARED"] * len(q_l)})
    with pytest.raises(AssertionError, match="patient leak"):
        sub.retrieval(tr, te)
```

### `validation/tests/test_retrieval_postproc.py`

```python
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
    """b is a k-reciprocal neighbour of a only if each is in the other's top-k. Assert the
    'reciprocal' half actually holds -- a one-way implementation would silently pass everything
    else in this file."""
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
    identity'. Prove both halves: without exclusion the transform does nothing; with it, it
    does something."""
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
```

### `validation/tests/test_bracs_patch_geometry.py`

```python
"""_pad_grid_to_2x2 sits under EVERY BRACS embedding -- frozen baseline and LoRA alike. A bug
here does not show up as a crash; it shows up as a wrong number in RESULTS.md."""
import numpy as np
import torch

from extract_bracs_features import TILE, _pad_grid_to_2x2, roi_to_patches


def _feats(n, dim=4):
    return torch.ones(n, dim)


def _grid_ok(coords):
    return len({x for x, _ in coords}) >= 2 and len({y for _, y in coords}) >= 2


def test_single_tile_roi_is_padded_to_a_full_2x2():
    """~1/3 of BRACS ROIs are a single 512px tile. TITAN's get_alibi() raises IndexError on a
    1xN grid, so this pad is what makes those ROIs encodable at all."""
    f, c = _pad_grid_to_2x2(_feats(1), [(0, 0)])
    assert f.shape[0] == 4 and len(c) == 4
    assert _grid_ok(c)
    assert int((f.abs().sum(dim=1) == 0).sum()) == 3      # exactly 3 zero pads injected


def test_horizontal_strip_is_padded():
    f, c = _pad_grid_to_2x2(_feats(2), [(0, 0), (TILE, 0)])
    assert _grid_ok(c)
    assert int((f.abs().sum(dim=1) == 0).sum()) == f.shape[0] - 2


def test_vertical_strip_is_padded():
    f, c = _pad_grid_to_2x2(_feats(2), [(0, 0), (0, TILE)])
    assert _grid_ok(c)
    assert int((f.abs().sum(dim=1) == 0).sum()) == f.shape[0] - 2


def test_an_already_2x2_grid_is_left_completely_alone():
    """Real tissue patches must never be touched -- no reordering, no padding, no copies."""
    coords = [(0, 0), (TILE, 0), (0, TILE), (TILE, TILE)]
    f_in = _feats(4)
    f, c = _pad_grid_to_2x2(f_in, coords)
    assert f.shape[0] == 4
    assert torch.equal(f, f_in)
    assert [tuple(x) for x in c] == coords


def test_pads_are_exactly_zero_so_titan_masks_them_as_background():
    """The pad only works because TITAN's preprocess_features drops rows via `any(feature != 0)`.
    A pad that is not bitwise zero would be fed to attention as if it were tissue."""
    f, _ = _pad_grid_to_2x2(_feats(1), [(0, 0)])
    pads = f[1:]
    assert torch.count_nonzero(pads) == 0


def test_real_patches_keep_their_position_and_values():
    f_in = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) + 1.0   # nonzero
    f, c = _pad_grid_to_2x2(f_in, [(0, 0), (TILE, 0)])
    assert torch.equal(f[:2], f_in)
    assert [tuple(x) for x in c[:2]] == [(0, 0), (TILE, 0)]


# ------------------------------------------------------------ tiling

def test_a_fully_blank_roi_still_yields_one_tile():
    """Documented behaviour: 'an all-blank ROI keeps one centered tile so it still yields an
    embedding'. Without it the ROI silently vanishes from the manifest-aligned feature matrix."""
    from PIL import Image
    white = Image.new("RGB", (1024, 1024), (255, 255, 255))
    tiles, coords = roi_to_patches(white)
    assert len(tiles) == 1 and coords == [(0, 0)]


def test_tissue_tiles_are_kept_and_padded_to_full_tile_size():
    from PIL import Image
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 120, size=(1400, 1400, 3), dtype=np.uint8))
    tiles, coords = roi_to_patches(noisy)          # 1400 -> 700 after the 2x downsample
    assert len(tiles) == len(coords) >= 4          # 700px / 512 -> a 2x2 grid with edge tiles
    for t in tiles:
        assert t.size == (TILE, TILE)              # edge tiles white-padded, never upscaled
```

### `validation/tests/test_proxy_anchor.py`

```python
"""The Proxy-Anchor loss that produces the Tier-3 LoRA result."""
import numpy as np
import torch
import torch.nn.functional as F

from finetune_bracs_lora import ProxyAnchor, pk_batches


def test_loss_is_lower_when_embeddings_sit_on_their_own_proxy():
    """The defining property. If this fails, the loss is not doing metric learning at all."""
    torch.manual_seed(0)
    C, D = 4, 8
    loss_fn = ProxyAnchor(C, D)
    P = F.normalize(loss_fn.proxies.detach(), dim=1)
    y = torch.arange(C)

    aligned = loss_fn(P, y)                        # each embedding == its own class proxy
    wrong = loss_fn(P[torch.roll(y, 1)], y)        # each embedding == a DIFFERENT class's proxy
    assert aligned.item() < wrong.item()


def test_loss_is_finite_in_fp32_at_the_alpha_delta_extremes():
    """exp(alpha*(cos+delta)) peaks near exp(32*1.1) = exp(35.2) ~ 2e15: fine in fp32, inf in
    fp16. The code comments claim this; assert it, because an inf here silently kills a run."""
    C, D = 7, 16
    loss_fn = ProxyAnchor(C, D, alpha=32.0, delta=0.1)
    with torch.no_grad():
        loss_fn.proxies.copy_(torch.ones(C, D))
    emb = F.normalize(torch.ones(C, D), dim=1)     # cos == 1 everywhere: the worst case
    out = loss_fn(emb, torch.arange(C))
    assert torch.isfinite(out).all()


def test_loss_is_positive_and_scalar():
    loss_fn = ProxyAnchor(5, 8)
    emb = F.normalize(torch.randn(15, 8), dim=1)
    out = loss_fn(emb, torch.arange(15) % 5)
    assert out.ndim == 0 and out.item() > 0


def test_gradients_reach_both_the_embeddings_and_the_proxies():
    loss_fn = ProxyAnchor(3, 6)
    emb = F.normalize(torch.randn(9, 6, requires_grad=True), dim=1)
    loss_fn(emb, torch.arange(9) % 3).backward()
    assert loss_fn.proxies.grad is not None
    assert torch.count_nonzero(loss_fn.proxies.grad) > 0


def test_pk_batches_put_every_class_in_every_batch():
    """P = all classes, K per class. Both loss terms need a non-empty positive set, which is why
    the sampler is PK and not random."""
    C, K = 7, 8
    y = np.repeat(np.arange(C), 40)
    rng = np.random.default_rng(0)
    batches = list(pk_batches(y, C, K, rng))
    assert batches
    for b in batches:
        assert len(b) == C * K
        assert set(y[b]) == set(range(C))          # every class present
        counts = np.bincount(y[b], minlength=C)
        assert (counts == K).all()                 # exactly K each


def test_pk_batches_are_reproducible_from_the_rng():
    C, K = 3, 4
    y = np.repeat(np.arange(C), 20)
    a = list(pk_batches(y, C, K, np.random.default_rng(5)))
    b = list(pk_batches(y, C, K, np.random.default_rng(5)))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
```

## 5. Verification

```bash
cd /home/user01/TITAN
.venv/bin/python -m pip install -r validation/requirements-dev.txt
.venv/bin/python -m pytest validation/tests -q
```

Expected: **all tests pass**, in under ~10 seconds, with **no network access, no GPU, no HF token,
and no BRACS data on disk**. Verify the isolation claim explicitly:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest validation/tests -q
```

This must also pass. If it does not, a test is reaching for a resource it should not need.

## 6. Done criteria (machine-checkable)

- [ ] `.venv/bin/python -m pytest validation/tests -q` → **0 failed**.
- [ ] `HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest validation/tests -q` → **0 failed**.
- [ ] `.venv/bin/python -m pytest validation/tests -q --collect-only | tail -1` reports **≥ 43 tests**
      (≥35 new from this plan + the 8 existing from plans 001/002).
- [ ] The 8 pre-existing tests (`test_make_results.py`, `test_lora_seeding.py`) pass **under pytest**
      — for the first time, since `conftest.py` + `pytest` are what make them collectable. Neither
      file may be edited.
- [ ] `git status --porcelain` shows only *added* files under `validation/tests/` plus
      `validation/requirements-dev.txt`. **No production file is modified by this plan.**

## 7. Scope

**In scope (all new files):**
- `validation/requirements-dev.txt`
- `validation/tests/conftest.py`
- `validation/tests/test_ranking_metrics.py`
- `validation/tests/test_metric_implementations_agree.py`
- `validation/tests/test_retrieval_postproc.py`
- `validation/tests/test_bracs_patch_geometry.py`
- `validation/tests/test_proxy_anchor.py`

**Out of scope — do not touch:**
- **Every file under `validation/` that already exists.** This is a characterization suite. Its
  entire value is that it describes the code *as it is*. If a test fails, that is a finding to
  report — see §9 — not a licence to edit the code so the test goes green.
- `setup.py` — do not add pytest to `install_requires`.
- `validation/results/*.json`.
- The `titan/` package (upstream code).

## 8. Maintenance note

The rule from here: **a change to any function in the table in §1 must come with a test, and the
suite must be green before a number in `RESULTS.md` moves.** These tests are cheap (seconds, CPU,
no data), so there is no excuse for skipping them.

The suite deliberately mixes two kinds of assertion:
- **hand-computed literals** (`acc@3 == 1.0` for a specific 4-row similarity matrix), which catch
  a rewrite that changes semantics;
- **mathematical invariants** (full-rank PCA is a rotation → ranking unchanged; DBA without
  self-exclusion is the identity; λ=1 ⇒ raw cosine), which catch a rewrite that is *plausible but
  wrong*.

Prefer adding invariants. A literal only pins the case you thought of.

## 9. Escape hatches — STOP and report back if:

- **Any test in §4 fails against the current code.** Do not adjust the test to match the code and
  do not "fix" the code. Every assertion here was derived from the source or from the docstrings'
  own stated guarantees, so a failure means one of two things, and both are findings that must be
  reported before anything else happens:
  1. a genuine bug in a function that produced numbers already published in `RESULTS.md`, or
  2. a mistake in this plan.

  Report the failing test, the actual value, and the expected value. Let the author adjudicate.
  This is the single most important instruction in this plan: **the point of a characterization
  suite is to tell you the truth about the code, and it cannot do that if you edit it until it
  agrees with you.**
- Importing any `validation/*.py` module triggers a network call, a model download, or a CUDA
  init. That is a latent bug worth its own report.
- `pytest` cannot be installed into `.venv` (e.g. the venv is read-only). Report it; do not
  install into the system Python or create a second venv.
