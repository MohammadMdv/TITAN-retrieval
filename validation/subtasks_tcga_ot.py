"""Further TITAN tests on the free TCGA_TITAN_features.pkl: harder patient-disjoint
sub-typing tasks (linear probe + retrieval). Zero download. Splits (train/val/test) are
the TCGA-OT splits, already patient-disjoint; each task just filters by OncoTreeCode.

For every task we report:
  - Linear probe balanced accuracy (parallel C-sweep, C picked on val log-loss; == paper method)
  - Retrieval Acc@1 / MVAcc@3 (DB = train, query = test, cosine)
  - patient (case_id) disjointness is asserted.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from common import load_hf_token, DATASETS, save_results

# task -> list of OncoTreeCodes (a subtype problem within one organ system)
TASKS = {
    "NSCLC_LUAD_vs_LUSC":      ["LUAD", "LUSC"],
    "RCC_3way":                ["CCRCC", "PRCC", "CHRCC"],
    "BRCA_IDC_vs_ILC":         ["IDC", "ILC"],
    "Sarcoma_4way":            ["MFH", "MFS", "LMS", "DDLS"],
}
# brain: GBM (grade IV) vs lower-grade glioma -> handled specially (label collapse)
BRAIN_GBM = ["GBM"]
BRAIN_LGG = ["OAST", "ODG", "AASTR", "AOAST", "ASTR"]


def load_emb_df():
    from huggingface_hub import hf_hub_download
    load_hf_token()
    p = hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl")
    with open(p, "rb") as f:
        d = pickle.load(f)
    return pd.DataFrame({"slide_id": d["filenames"], "emb": list(np.asarray(d["embeddings"]))})


def splits_with_labels(emb_df):
    frames = []
    for s in ["train", "val", "test"]:
        c = pd.read_csv(DATASETS / f"tcga-ot_{s}.csv").assign(split=s)
        frames.append(pd.merge(emb_df, c, on="slide_id"))
    return pd.concat(frames, ignore_index=True)


def subset(df, codes, label_col="OncoTreeCode"):
    return df[df[label_col].isin(codes)].copy()


def linear_probe(tr, va, te, label_col="_y"):
    from joblib import Parallel, delayed
    from sklearn.preprocessing import LabelEncoder
    from titan.utils import get_eval_metrics
    from classification_tcga_ot import _fit_one

    le = LabelEncoder().fit(pd.concat([tr[label_col], va[label_col], te[label_col]]))
    def XY(d): return np.stack(d["emb"].values), le.transform(d[label_col])
    Xtr, ytr = XY(tr); Xva, yva = XY(va); Xte, yte = XY(te)
    grid = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    fits = Parallel(n_jobs=int(os.environ.get("N_JOBS", 10)))(
        delayed(_fit_one)(c, Xtr, ytr, Xva, yva, 500) for c in grid)
    model = fits[int(np.argmin([v for v, _ in fits]))][1]
    preds = model.predict(Xte); probs = model.predict_proba(Xte)
    roc = {"multi_class": "ovo", "average": "macro"} if len(le.classes_) > 2 else {}
    res = get_eval_metrics(yte, preds, probs, roc_kwargs=roc)
    return {k.strip("/"): float(v) for k, v in res.items() if isinstance(v, (int, float))}


def retrieval(tr, te, label_col="_y"):
    Xdb = np.stack(tr["emb"].values).astype(np.float32); db_lab = tr[label_col].values
    Xq = np.stack(te["emb"].values).astype(np.float32); q_lab = te[label_col].values
    assert set(tr["case_id"]).isdisjoint(te["case_id"]), "patient leak in retrieval"
    Xdb /= (np.linalg.norm(Xdb, axis=1, keepdims=True) + 1e-8)
    Xq /= (np.linalg.norm(Xq, axis=1, keepdims=True) + 1e-8)
    sim = Xq @ Xdb.T
    a1 = mv3 = 0
    for i in range(len(q_lab)):
        order = np.argsort(-sim[i])[:3]
        retr = db_lab[order]
        a1 += retr[0] == q_lab[i]
        mv3 += Counter(retr).most_common(1)[0][0] == q_lab[i]
    return {"acc@1": a1 / len(q_lab), "mvacc@3": mv3 / len(q_lab)}


def run_task(name, df):
    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    # patient-disjoint guardrail
    assert set(tr.case_id).isdisjoint(te.case_id) and set(tr.case_id).isdisjoint(va.case_id)
    lp = linear_probe(tr, va, te)
    rt = retrieval(tr, te)
    info = {
        "classes": sorted(df["_y"].unique().tolist()),
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "test_class_counts": dict(Counter(te["_y"])),
        "linear_probe": lp, "retrieval": rt,
    }
    print(f"\n### {name}  classes={info['classes']}")
    print(f"    n tr/va/te = {info['n']['train']}/{info['n']['val']}/{info['n']['test']}  "
          f"test dist={info['test_class_counts']}")
    print(f"    LP bacc={lp.get('bacc'):.4f}  auroc={lp.get('auroc')}  | "
          f"Retrieval Acc@1={rt['acc@1']:.4f}  MVAcc@3={rt['mvacc@3']:.4f}")
    return info


def main():
    emb_df = load_emb_df()
    full = splits_with_labels(emb_df)

    results = {}
    for name, codes in TASKS.items():
        d = subset(full, codes)
        d["_y"] = d["OncoTreeCode"]
        results[name] = run_task(name, d)

    # brain: GBM vs LGG (label collapse)
    d = subset(full, BRAIN_GBM + BRAIN_LGG)
    d["_y"] = np.where(d["OncoTreeCode"].isin(BRAIN_GBM), "GBM", "LGG")
    results["Brain_GBM_vs_LGG"] = run_task("Brain_GBM_vs_LGG", d)

    print("\n==================== SUMMARY ====================")
    print(f"{'task':26s}{'#cls':>5s}{'test_n':>7s}{'LP_bacc':>9s}{'Ret@1':>8s}{'MV@3':>7s}")
    for name, r in results.items():
        print(f"{name:26s}{len(r['classes']):>5d}{r['n']['test']:>7d}"
              f"{r['linear_probe'].get('bacc'):>9.4f}{r['retrieval']['acc@1']:>8.4f}"
              f"{r['retrieval']['mvacc@3']:>7.4f}")

    save_results("subtasks_tcga_ot.json", results)
    print("\n[subtasks] OK")


if __name__ == "__main__":
    sys.exit(main())
