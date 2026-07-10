"""TITAN-based slide retrieval on CAMELYON16 — PathSearch-style, for direct comparison.

Replicates PathSearch's exact protocol on the same 100-slide subset:
  - queries  = the same 30 slides PathSearch used (from camelyon16_query_results.csv)
  - database = the remaining 70 slides (disjoint; verified PathSearch retrieved only from these)
  - metrics  = Top-1 acc, Top-3-MV, Top-5-MV, overall + per class

Uses TITAN slide embeddings (cached from Phase 2) with cosine similarity (TITAN's standard,
CLIP-style normalized space). Also reports Euclidean (PathSearch's metric) for reference.
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
import torch

from common import RESULTS_DIR, save_results

# CSV of {query_id, query_label} for the external PathSearch comparison run -- not part of
# this repo. Only needed if you're reproducing that specific comparison.
QR = os.environ.get("PATHSEARCH_QUERY_CSV", "")
CACHE = RESULTS_DIR / "cam16_titan_emb.pt"

# PathSearch reference numbers on this exact subset (provided by user)
PATHSEARCH = {
    "overall": {"top1": 0.8667, "top3_mv": 0.8000, "top5_mv": 0.7667},
    "normal":  {"top1": 1.0000, "top3_mv": 1.0000, "top5_mv": 1.0000},
    "tumor":   {"top1": 0.6667, "top3_mv": 0.5000, "top5_mv": 0.4167},
}


def load_queries():
    q = {}
    with open(QR) as f:
        for r in csv.DictReader(f):
            q[r["query_id"]] = r["query_label"]
    return q


def mv_correct(labels_topk, true):
    """Majority vote of top-k == true (odd k -> no ties)."""
    return Counter(labels_topk).most_common(1)[0][0] == true


def evaluate(sim, db_labels, q_labels, higher_is_better=True):
    """sim: queries x db. Returns overall + per-class Top-1 / Top-3-MV / Top-5-MV."""
    order = np.argsort(-sim if higher_is_better else sim, axis=1)  # best first
    rows = {"top1": [], "top3_mv": [], "top5_mv": []}
    per_class = {c: {"top1": [], "top3_mv": [], "top5_mv": []} for c in set(q_labels)}
    for i, true in enumerate(q_labels):
        ranked = db_labels[order[i]]
        r1 = ranked[0] == true
        r3 = mv_correct(ranked[:3], true)
        r5 = mv_correct(ranked[:5], true)
        for key, val in zip(("top1", "top3_mv", "top5_mv"), (r1, r3, r5)):
            rows[key].append(val)
            per_class[true][key].append(val)
    overall = {k: float(np.mean(v)) for k, v in rows.items()}
    per_class = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in per_class.items()}
    return overall, per_class


def main():
    if not QR:
        print("[demo] set PATHSEARCH_QUERY_CSV to the PathSearch query-results CSV to run this comparison."); return 1
    if not CACHE.exists():
        print(f"[demo] missing {CACHE}; run classification_camelyon16.py first."); return 1
    d = torch.load(CACHE)
    emb = d["emb"].numpy().astype(np.float32)          # (100, 768)
    ids = np.array(d["ids"])
    id2emb = {s: emb[i] for i, s in enumerate(ids)}

    queries = load_queries()
    q_ids = [s for s in ids if s in queries]
    db_ids = [s for s in ids if s not in queries]
    assert set(q_ids).isdisjoint(db_ids)
    Xq = np.stack([id2emb[s] for s in q_ids]); q_labels = np.array([queries[s] for s in q_ids])
    # database labels from filename prefix mapping via reference.csv-consistent scheme:
    # reuse Phase 2's cached label semantics — infer from the same reference file
    import classification_camelyon16 as p2
    ref = p2.load_labels()
    Xdb = np.stack([id2emb[s] for s in db_ids]); db_labels = np.array([ref[s] for s in db_ids])

    print(f"[demo] queries={len(q_ids)} ({Counter(q_labels)})  database={len(db_ids)} ({Counter(db_labels)})")

    # cosine (TITAN standard): L2-normalize then inner product
    Xq_n = Xq / (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-8)
    Xdb_n = Xdb / (np.linalg.norm(Xdb, axis=1, keepdims=True) + 1e-8)
    cos = Xq_n @ Xdb_n.T
    cos_overall, cos_per = evaluate(cos, db_labels, q_labels, higher_is_better=True)

    # euclidean (PathSearch's metric) for reference: distance -> lower is better
    d2 = ((Xq[:, None, :] - Xdb[None, :, :]) ** 2).sum(-1)
    euc_overall, euc_per = evaluate(d2, db_labels, q_labels, higher_is_better=False)

    def show(name, overall, per):
        print(f"\n=== TITAN retrieval ({name}) ===")
        print(f"  overall  Top-1={overall['top1']:.4f}  Top-3-MV={overall['top3_mv']:.4f}  Top-5-MV={overall['top5_mv']:.4f}")
        for c in ("normal", "tumor"):
            p = per[c]
            print(f"  {c:7s} Top-1={p['top1']:.4f}  Top-3-MV={p['top3_mv']:.4f}  Top-5-MV={p['top5_mv']:.4f}")

    show("cosine", cos_overall, cos_per)
    show("euclidean", euc_overall, euc_per)

    print("\n=== vs PathSearch (same 30 queries / 70 DB) ===")
    print(f"  {'':12s}{'TITAN-cos':>10s}{'TITAN-euc':>10s}{'PathSearch':>12s}")
    for metric in ("top1", "top3_mv", "top5_mv"):
        print(f"  overall {metric:8s}{cos_overall[metric]:>10.4f}{euc_overall[metric]:>10.4f}{PATHSEARCH['overall'][metric]:>12.4f}")
    for c in ("normal", "tumor"):
        for metric in ("top1", "top3_mv", "top5_mv"):
            print(f"  {c:6s} {metric:8s}{cos_per[c][metric]:>10.4f}{euc_per[c][metric]:>10.4f}{PATHSEARCH[c][metric]:>12.4f}")

    save_results("retrieval_camelyon16.json", {
        "n_queries": len(q_ids), "n_database": len(db_ids),
        "titan_cosine": {"overall": cos_overall, "per_class": cos_per},
        "titan_euclidean": {"overall": euc_overall, "per_class": euc_per},
        "pathsearch": PATHSEARCH,
    })
    print("\n[demo] OK")


if __name__ == "__main__":
    sys.exit(main())
