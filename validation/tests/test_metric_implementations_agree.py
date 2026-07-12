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
import pytest

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
    """subtasks builds its own cosine from embeddings, so this pins the whole path."""
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
    Xdb, db_l, Xq, q_l = _synthetic()
    tr = pd.DataFrame({"emb": list(Xdb), "_y": db_l, "case_id": ["SHARED"] * len(db_l)})
    te = pd.DataFrame({"emb": list(Xq), "_y": q_l, "case_id": ["SHARED"] * len(q_l)})
    with pytest.raises(AssertionError, match="patient leak"):
        sub.retrieval(tr, te)
