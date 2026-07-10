"""Shared retrieval helpers: data loading, the patient-disjoint database, and ranking metrics.

Used by retrieval_tcga_ot.py (raw cosine baseline) and the Tier-1 post-hoc experiments
(whitening / k-reciprocal re-ranking / query expansion). None of these train anything --
TITAN's embeddings are frozen and untouched; only the *search procedure* changes.

Protocol (identical across every Tier-1 experiment, so deltas isolate the technique):
  database  = train split, minus any slide whose case_id appears in val or test
  val queries  = val split   -> used ONLY to pick hyperparameters
  test queries = test split  -> used ONLY once, to report the selected config
The database is fixed and patient-disjoint from both query sets, so a config tuned on val
transfers to test without the database changing underneath it.
"""
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from common import load_hf_token, DATASETS

# Raw-cosine baseline on this exact protocol (see retrieval_tcga_ot.py); paper refs 0.880/0.807.
BASELINE_REF = {"acc@3": 0.8717, "mvacc@3": 0.7812}
K = 3


def load_embeddings():
    from huggingface_hub import hf_hub_download
    load_hf_token()
    path = hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return pd.DataFrame({"slide_id": data["filenames"],
                         "embeddings": list(np.asarray(data["embeddings"]))})


def build_split(emb_df, name, target="OncoTreeCode"):
    s = pd.read_csv(DATASETS / f"tcga-ot_{name}.csv")
    m = pd.merge(emb_df, s, on="slide_id")
    X = np.stack(m["embeddings"].values).astype(np.float32)
    return X, m[target].values, m["case_id"].values, m["slide_id"].values


def l2_normalize(X, eps=1e-8):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def load_retrieval_data():
    """Returns (Xdb, db_labels, Xval, val_labels, Xte, te_labels), all raw (un-normalized).

    The database drops train slides whose patient appears in val or test -- the same
    leakage guard retrieval_tcga_ot.py applies, asserted here too.
    """
    emb_df = load_embeddings()
    Xtr, ytr, ctr, _ = build_split(emb_df, "train")
    Xva, yva, cva, _ = build_split(emb_df, "val")
    Xte, yte, cte, _ = build_split(emb_df, "test")

    forbidden = set(cva) | set(cte)
    keep = np.array([c not in forbidden for c in ctr])
    Xdb, db_labels, db_cases = Xtr[keep], ytr[keep], ctr[keep]

    assert set(db_cases).isdisjoint(set(cte)), "patient leakage: test case in database"
    assert set(db_cases).isdisjoint(set(cva)), "patient leakage: val case in database"

    print(f"[data] db={len(db_labels)} (dropped {int((~keep).sum())} leaking) "
          f"val_q={len(yva)} test_q={len(yte)}")
    return Xdb, db_labels, Xva, yva, Xte, yte


def eval_ranking(sim, db_labels, q_labels, k=K):
    """sim: (n_queries, n_db), higher = more similar. Returns acc@1, acc@k, mvacc@k.

    acc@k   = at least one of the top-k shares the query's label.
    mvacc@k = majority label among top-k equals the query's label (ties -> highest similarity).
    """
    acc1 = acc_k = mv_k = 0
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])[:k]
        retr = db_labels[order]
        acc1 += retr[0] == q_labels[i]
        acc_k += q_labels[i] in set(retr)
        counts = Counter(retr)
        top = max(counts.values())
        cands = [lab for lab in retr if counts[lab] == top]  # retr is in descending-sim order
        mv_k += cands[0] == q_labels[i]
    n = sim.shape[0]
    return {"acc@1": acc1 / n, f"acc@{k}": acc_k / n, f"mvacc@{k}": mv_k / n}


def cosine_sim(Xq, Xdb):
    return l2_normalize(Xq) @ l2_normalize(Xdb).T


def baseline_ranking(Xdb, db_labels, Xq, q_labels):
    """Raw cosine on the frozen embeddings -- the number every experiment must beat."""
    return eval_ranking(cosine_sim(Xq, Xdb), db_labels, q_labels)


def fmt_metrics(m):
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())


def print_delta_table(title, baseline, rows, select_metric):
    """rows: list of (label, metrics_dict). Prints each row's delta vs baseline."""
    keys = list(baseline.keys())
    print(f"\n{'='*len(title)}\n{title}\n{'='*len(title)}")
    header = f"  {'config':<34s}" + "".join(f"{k:>10s}{'Δ':>9s}" for k in keys)
    print(header)
    print(f"  {'baseline (raw cosine)':<34s}" + "".join(f"{baseline[k]:>10.4f}{'—':>9s}" for k in keys))
    for label, m in rows:
        cells = "".join(f"{m[k]:>10.4f}{m[k]-baseline[k]:>+9.4f}" for k in keys)
        star = " *" if m[select_metric] > baseline[select_metric] else ""
        print(f"  {label:<34s}{cells}{star}")
