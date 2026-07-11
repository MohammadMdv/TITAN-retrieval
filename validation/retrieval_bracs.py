"""BRACS ROI retrieval baseline (patient-disjoint), raw cosine on frozen TITAN embeddings.

This is the number the LoRA/adapter study has to beat. Nothing is trained.

Protocol (mirrors retrieval_tcga_ot.py, but on the BRACS ROI features):
  database     = train split (3,657 ROIs)
  test queries = test split  (570 ROIs)   -- the headline number
  val queries  = val split   (312 ROIs)   -- reported too, for later hyperparameter tuning
The BRACS ROI split is already patient-disjoint on all three pairs (verified at download);
asserted again here on the packed features so a corrupt pickle cannot slip through.

Metrics (7 lesion classes): Acc@1, Acc@3, MVAcc@3, Acc@5, MVAcc@5, plus per-class Acc@1.
Two chance references are printed for context: retrieving a uniformly random database item
(expected Acc@1 = sum_c p_query(c) p_db(c)), and always guessing the most common class.

Input : data/BRACS/features/bracs_roi_titan.pkl (from extract_bracs_features.py).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from collections import Counter

import numpy as np
import pandas as pd

from common import save_results, TITAN_ROOT
from retrieval_common import l2_normalize, eval_ranking, cosine_sim

FEAT_PKL = TITAN_ROOT / "data" / "BRACS" / "features" / "bracs_roi_titan.pkl"


def load_split(df, split):
    d = df[df["split"] == split]
    X = np.stack(d["embedding"].values).astype(np.float32)
    return X, d["label"].values, d["patient_id"].values, d["item_id"].values


def per_class_acc1(sim, db_labels, q_labels):
    """Acc@1 broken down by the query's true class."""
    top1 = db_labels[np.argmax(sim, axis=1)]
    out = {}
    for c in sorted(set(q_labels)):
        m = q_labels == c
        out[c] = float((top1[m] == q_labels[m]).mean())
    return out


def chance_references(db_labels, q_labels):
    n_db, n_q = len(db_labels), len(q_labels)
    pdb = {c: n / n_db for c, n in Counter(db_labels).items()}
    pq = {c: n / n_q for c, n in Counter(q_labels).items()}
    random_acc1 = sum(pq.get(c, 0) * pdb.get(c, 0) for c in set(pdb) | set(pq))
    majority_acc1 = max(pq.values())
    return {"random_retrieval_acc@1": random_acc1, "majority_class_acc@1": majority_acc1}


def evaluate(Xdb, db_labels, db_cases, Xq, q_labels, q_cases, name):
    assert set(db_cases).isdisjoint(set(q_cases)), f"patient leak: db vs {name}"
    sim = cosine_sim(Xq, Xdb)
    m3 = eval_ranking(sim, db_labels, q_labels, k=3)
    m5 = eval_ranking(sim, db_labels, q_labels, k=5)
    metrics = {"acc@1": m3["acc@1"], "acc@3": m3["acc@3"], "mvacc@3": m3["mvacc@3"],
               "acc@5": m5["acc@5"], "mvacc@5": m5["mvacc@5"]}
    per_class = per_class_acc1(sim, db_labels, q_labels)
    return metrics, per_class


def main():
    sys.stdout.reconfigure(line_buffering=True)
    if not FEAT_PKL.exists():
        sys.exit(f"[ret-bracs] {FEAT_PKL} not found -- run extract_bracs_features.py first.")

    df = pd.read_pickle(FEAT_PKL)
    Xtr, ytr, ctr, _ = load_split(df, "train")
    Xva, yva, cva, _ = load_split(df, "val")
    Xte, yte, cte, _ = load_split(df, "test")
    print(f"[ret-bracs] db(train)={len(ytr)}  val_q={len(yva)}  test_q={len(yte)}  "
          f"classes={sorted(set(ytr))}")

    chance = chance_references(ytr, yte)
    print(f"[ret-bracs] chance: random-retrieval Acc@1={chance['random_retrieval_acc@1']:.4f}  "
          f"majority-class Acc@1={chance['majority_class_acc@1']:.4f}")

    results = {"n": {"db": len(ytr), "val": len(yva), "test": len(yte)},
               "classes": sorted(set(ytr)), "chance": chance}

    for split_name, (Xq, yq, cq) in [("val", (Xva, yva, cva)), ("test", (Xte, yte, cte))]:
        metrics, per_class = evaluate(Xtr, ytr, ctr, Xq, yq, cq, split_name)
        results[split_name] = {"metrics": metrics, "per_class_acc@1": per_class}
        print(f"\n[ret-bracs] === {split_name.upper()} queries (n={len(yq)}) ===")
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print("  per-class Acc@1: " + "  ".join(f"{c}={v:.3f}" for c, v in per_class.items()))

    save_results("retrieval_bracs.json", results)
    print("\n[ret-bracs] baseline saved. This is the number LoRA/adapters must beat (test split).")
    print("[ret-bracs] OK")


if __name__ == "__main__":
    sys.exit(main())
