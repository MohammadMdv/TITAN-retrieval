"""Phase 9: before/after eval of the SupCon-fine-tuned projection head (Phase 8) against the
exact protocol already used for TCGA-OT's established baselines:
  phase3_tcga_ot.py   linear-probe bacc  = 0.7026 (paper ref 0.704)
  phase4_retrieval.py Acc@3 / MVAcc@3    = 0.8717 / 0.7812 (paper ref 0.880 / 0.807)

BEFORE = raw frozen 768-d TITAN embeddings (identical inputs to phase3/phase4).
AFTER  = those same embeddings passed through the frozen, SupCon-trained 256-d projection
         head (best-val-loss checkpoint from phase8_finetune_supcon.py).

Same protocol both times, so the delta isolates the effect of the head:
  - Linear probe: 45-point C sweep (paper grid), best C by val log_loss, parallel across cores
    (reuses phase3_tcga_ot._fit_one exactly).
  - Retrieval: DB=train (val/test case_id leakage dropped, matching phase4), queries=test,
    cosine, Acc@3 / MVAcc@3.
  - Patient-disjoint asserted both times, on both passes.

The test split is untouched by phase8 (SupCon only ever sees train, model selection only ever
sees val) -- this is the first and only time test is used.

Requires phase8_finetune_supcon.py to have been run first (reads ssl_head_best.pt).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from collections import Counter

import numpy as np
import torch
import yaml

from common import get_device, save_results, RESULTS_DIR, CONFIG_TCGA_OT
from ssl_common import load_embeddings, build_split, ProjHead, PROJ_DIM
from phase3_tcga_ot import _fit_one  # identical C-sweep fit routine as the established baseline

CKPT_BEST = RESULTS_DIR / "ssl_head_best.pt"
REF = {"lp_bacc": 0.7026, "acc@3": 0.8717, "mvacc@3": 0.7812}


def linear_probe(Xtr, ytr, Xva, yva, Xte, yte):
    from joblib import Parallel, delayed
    from titan.utils import get_eval_metrics
    grid = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)  # paper grid
    n_jobs = int(os.environ.get("N_JOBS", 10))
    fits = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(c, Xtr, ytr, Xva, yva, 500) for c in grid)
    best_i = int(np.argmin([vl for vl, _ in fits]))
    model = fits[best_i][1]
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)
    res = get_eval_metrics(yte, preds, probs, roc_kwargs={"multi_class": "ovo", "average": "macro"})
    return {k.strip("/"): float(v) for k, v in res.items() if isinstance(v, (int, float))}


def topk_metrics(sim, db_labels, q_labels, K=3):
    acc_hits, mv_hits = 0, 0
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])[:K]
        retr = db_labels[order]
        if q_labels[i] in set(retr):
            acc_hits += 1
        counts = Counter(retr)
        top = max(counts.values())
        cands = [lab for lab in retr if counts[lab] == top]  # keeps descending-sim order
        if cands[0] == q_labels[i]:
            mv_hits += 1
    return acc_hits / sim.shape[0], mv_hits / sim.shape[0]


def retrieval(Xtr, ytr_raw, ctr, Xte, yte_raw, cte, cva, K=3):
    """Mirrors phase4_retrieval.py exactly: DB=train with val/test case_id leakage dropped."""
    forbidden = set(cte) | set(cva)
    keep = np.array([c not in forbidden for c in ctr])
    Xdb, db_labels, db_cases = Xtr[keep], ytr_raw[keep], ctr[keep]
    assert set(db_cases).isdisjoint(set(cte)), "patient leakage: query case in database"
    Xdb_n = Xdb / (np.linalg.norm(Xdb, axis=1, keepdims=True) + 1e-8)
    Xq_n = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    sim = Xq_n @ Xdb_n.T
    acc3, mv3 = topk_metrics(sim, db_labels, yte_raw, K=K)
    return {"acc@3": acc3, "mvacc@3": mv3, "db_size": int(keep.sum())}


def evaluate(Xtr, ytr_raw, ctr, Xva, yva_raw, cva, Xte, yte_raw, cte, label_dict):
    ytr = np.array([label_dict[c] for c in ytr_raw])
    yva = np.array([label_dict[c] for c in yva_raw])
    yte = np.array([label_dict[c] for c in yte_raw])
    lp = linear_probe(Xtr, ytr, Xva, yva, Xte, yte)
    rt = retrieval(Xtr, ytr_raw, ctr, Xte, yte_raw, cte, cva)
    return lp, rt


@torch.no_grad()
def project(head, X, device):
    head.eval()
    return head(torch.from_numpy(X).to(device)).cpu().numpy().astype(np.float32)


def main():
    sys.stdout.reconfigure(line_buffering=True)  # visible progress even when piped/redirected
    with open(CONFIG_TCGA_OT) as f:
        label_dict = yaml.safe_load(f)["label_dict"]

    device = get_device()
    print("[phase9] loading cached embeddings + TCGA-OT splits ...")
    emb_df = load_embeddings()
    Xtr, ytr, ctr, idtr = build_split(emb_df, "train")
    Xva, yva, cva, idva = build_split(emb_df, "val")
    Xte, yte, cte, idte = build_split(emb_df, "test")
    print(f"[phase9] train={len(ytr)} val={len(yva)} test={len(yte)}")

    results = {}

    print(f"\n[phase9] === BEFORE (raw TITAN embeddings; ref bacc={REF['lp_bacc']} "
          f"acc@3={REF['acc@3']} mvacc@3={REF['mvacc@3']}) ===")
    lp_before, rt_before = evaluate(Xtr, ytr, ctr, Xva, yva, cva, Xte, yte, cte, label_dict)
    print(f"[phase9] BEFORE: LP bacc={lp_before['bacc']:.4f}  Acc@3={rt_before['acc@3']:.4f}  "
          f"MVAcc@3={rt_before['mvacc@3']:.4f}")
    results["before"] = {"linear_probe": lp_before, "retrieval": rt_before}
    save_results("phase9_eval_before_after.json", results)  # save BEFORE immediately

    if not CKPT_BEST.exists():
        print(f"\n[phase9] {CKPT_BEST} not found -- run phase8_finetune_supcon.py first, "
              f"then re-run this script for the AFTER comparison.")
        return 1

    head = ProjHead(Xtr.shape[1], PROJ_DIM).to(device)
    head.load_state_dict(torch.load(CKPT_BEST, map_location=device))
    print(f"\n[phase9] === AFTER (SupCon projection head, {CKPT_BEST.name}) ===")
    Ztr, Zva, Zte = project(head, Xtr, device), project(head, Xva, device), project(head, Xte, device)
    lp_after, rt_after = evaluate(Ztr, ytr, ctr, Zva, yva, cva, Zte, yte, cte, label_dict)
    print(f"[phase9] AFTER:  LP bacc={lp_after['bacc']:.4f}  Acc@3={rt_after['acc@3']:.4f}  "
          f"MVAcc@3={rt_after['mvacc@3']:.4f}")
    results["after"] = {"linear_probe": lp_after, "retrieval": rt_after}

    print("\n" + "=" * 60)
    print("[phase9] SUMMARY (before -> after)")
    print(f"  LP bacc: {lp_before['bacc']:.4f} -> {lp_after['bacc']:.4f}  "
          f"(delta {lp_after['bacc'] - lp_before['bacc']:+.4f})")
    print(f"  Acc@3:   {rt_before['acc@3']:.4f} -> {rt_after['acc@3']:.4f}  "
          f"(delta {rt_after['acc@3'] - rt_before['acc@3']:+.4f})")
    print(f"  MVAcc@3: {rt_before['mvacc@3']:.4f} -> {rt_after['mvacc@3']:.4f}  "
          f"(delta {rt_after['mvacc@3'] - rt_before['mvacc@3']:+.4f})")

    save_results("phase9_eval_before_after.json", results)
    print("[phase9] OK")


if __name__ == "__main__":
    sys.exit(main())
