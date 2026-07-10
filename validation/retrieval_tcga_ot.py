"""TCGA-OT slide retrieval (patient-disjoint).

Database = train split, queries = test split (already case_id-disjoint; asserted).
Metrics (K=3):
  Acc@K   = >=1 of top-K retrieved shares the query's OncoTreeCode.
  MVAcc@K = majority label of top-K equals query label (ties -> highest-similarity).

Targets: Acc@3 = 0.880, MVAcc@3 = 0.807.
"""
import sys
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import yaml

from common import load_hf_token, CONFIG_TCGA_OT, DATASETS, save_results


def load_embeddings():
    from huggingface_hub import hf_hub_download
    load_hf_token()
    path = hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return pd.DataFrame({"slide_id": data["filenames"],
                         "embeddings": list(np.asarray(data["embeddings"]))})


def build_split(emb_df, name, target):
    s = pd.read_csv(DATASETS / f"tcga-ot_{name}.csv")
    m = pd.merge(emb_df, s, on="slide_id")
    X = np.stack(m["embeddings"].values).astype(np.float32)
    labels = m[target].values
    cases = m["case_id"].values
    return X, labels, cases, m["slide_id"].values


def topk_metrics(sim, db_labels, q_labels, K=3):
    acc_hits, mv_hits = 0, 0
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])[:K]
        retr = db_labels[order]
        # Acc@K
        if q_labels[i] in set(retr):
            acc_hits += 1
        # MVAcc@K: most common; tie broken by rank (order is by descending sim)
        counts = Counter(retr)
        top = max(counts.values())
        cands = [lab for lab in retr if counts[lab] == top]  # keeps sim order
        mv = cands[0]
        if mv == q_labels[i]:
            mv_hits += 1
    return acc_hits / sim.shape[0], mv_hits / sim.shape[0]


def main():
    K = 3
    with open(CONFIG_TCGA_OT) as f:
        target = yaml.load(f, Loader=yaml.FullLoader)["target"]

    emb_df = load_embeddings()
    Xdb, db_labels, db_cases, db_ids = build_split(emb_df, "train", target)
    Xq, q_labels, q_cases, q_ids = build_split(emb_df, "test", target)
    _, _, val_cases, _ = build_split(emb_df, "val", target)

    # patient-disjoint guardrail: no query patient in the database (and drop val leakage too)
    forbidden = set(q_cases) | set(val_cases)
    keep = np.array([c not in forbidden for c in db_cases])
    n_drop = int((~keep).sum())
    Xdb, db_labels, db_cases, db_ids = Xdb[keep], db_labels[keep], db_cases[keep], db_ids[keep]
    assert set(db_cases).isdisjoint(set(q_cases)), "patient leakage: query case in database"
    print(f"[ret-tcga-ot] DB={len(db_ids)} (dropped {n_drop} leaking) queries={len(q_ids)}")

    # cosine similarity
    Xdb_n = Xdb / (np.linalg.norm(Xdb, axis=1, keepdims=True) + 1e-8)
    Xq_n = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-8)
    sim = Xq_n @ Xdb_n.T  # queries x db

    acc3, mv3 = topk_metrics(sim, db_labels, q_labels, K=K)
    print(f"[ret-tcga-ot] Acc@{K}={acc3:.4f} (ref 0.880)  MVAcc@{K}={mv3:.4f} (ref 0.807)")

    save_results("retrieval_tcga_ot.json", {
        "K": K, "db_size": len(db_ids), "n_queries": len(q_ids), "n_leaking_dropped": n_drop,
        "acc@3": acc3, "mvacc@3": mv3,
        "reference": {"acc@3": 0.880, "mvacc@3": 0.807},
        "patient_disjoint_asserted": True,
    })
    print("[ret-tcga-ot] OK")


if __name__ == "__main__":
    sys.exit(main())
