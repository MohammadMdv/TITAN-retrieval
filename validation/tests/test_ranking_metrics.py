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
    """Acc@k asks 'is the label anywhere in the top-k'; MVAcc@k asks 'does it win the vote'."""
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
