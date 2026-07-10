"""Tier-1 experiment 1/3: post-hoc whitening of frozen TITAN embeddings.

Nothing is trained. We fit a mean + covariance transform on the DATABASE embeddings only,
apply it to database and queries alike, L2-normalize, and search with cosine as before.
The frozen encoder is never touched, so there is no overfitting risk in the usual sense --
but the transform is still *estimated* from data, hence the val/test discipline below.

Why this might help: cosine similarity implicitly assumes the embedding's dimensions are
comparably scaled and uncorrelated. TITAN's 768-d space is neither -- a few high-variance
directions (often batch/stain/site effects rather than morphology) dominate the dot product.
Whitening equalizes the variance across directions, which can stop those few directions from
drowning out the discriminative ones.

Why it might hurt: whitening divides by sqrt(eigenvalue), so the smallest-variance directions
get amplified the most. Those are usually noise. Truncating to the top-d components before
whitening is the standard remedy, hence the dimension sweep.

Four transform families are compared, to separate "does dimensionality reduction help" from
"does whitening help" -- they are different claims and confounding them is a common mistake:
  center     : mean subtraction only (no rotation, no rescaling)
  pca-d      : center + project to top-d PCs, NO variance rescaling
  pcaw-d     : center + project to top-d PCs + whiten (divide by sqrt(eigenvalue))
  zca        : center + full-rank whitening, rotated back to the original axes

Hyperparameters (family, d) are selected on VAL queries and reported once on TEST queries.
Env: SELECT_METRIC (default acc@3), WHITEN_EPS (default 1e-6).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys

import numpy as np

from common import save_results
from retrieval_common import (load_retrieval_data, eval_ranking, l2_normalize,
                              baseline_ranking, print_delta_table, BASELINE_REF)

SELECT_METRIC = os.environ.get("SELECT_METRIC", "acc@3")
EPS = float(os.environ.get("WHITEN_EPS", 1e-6))
DIMS = [64, 128, 256, 384, 512, 768]



# Results are written per selection metric, so sweeps selected on a different metric
# (e.g. SELECT_METRIC=mvacc@3) do not overwrite the default acc@3 run.
def out_name(stem):
    tag = "" if SELECT_METRIC == "acc@3" else "_sel-" + SELECT_METRIC.replace("@", "")
    return f"{stem}{tag}.json"

def fit_pca(Xdb):
    """Eigendecomposition of the database covariance. Returns (mu, eigvals desc, eigvecs desc)."""
    mu = Xdb.mean(axis=0, keepdims=True)
    Xc = Xdb - mu
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    lam, V = np.linalg.eigh(cov)          # ascending, V columns are eigenvectors
    order = np.argsort(-lam)
    return mu, lam[order], V[:, order]


def make_transform(family, d, mu, lam, V, eps=EPS):
    """Returns f(X) -> transformed X. Fit quantities come from the database only."""
    if family == "center":
        return lambda X: X - mu
    if family == "pca":
        W = V[:, :d]
        return lambda X: (X - mu) @ W
    if family == "pcaw":
        W = V[:, :d] / np.sqrt(lam[:d] + eps)
        return lambda X: (X - mu) @ W
    if family == "zca":
        W = (V / np.sqrt(lam + eps)) @ V.T
        return lambda X: (X - mu) @ W
    raise ValueError(family)


def score(transform, Xdb, db_labels, Xq, q_labels):
    """Apply the transform, re-normalize, cosine-rank, evaluate."""
    Zdb = l2_normalize(transform(Xdb))
    Zq = l2_normalize(transform(Xq))
    return eval_ranking(Zq @ Zdb.T, db_labels, q_labels)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    Xdb, db_labels, Xva, yva, Xte, yte = load_retrieval_data()

    print(f"\n[whiten] fitting PCA on the {len(Xdb)} database embeddings only "
          f"(queries never contribute to mu/covariance)")
    mu, lam, V = fit_pca(Xdb)
    evr = lam / lam.sum()
    print(f"[whiten] top-10 explained variance ratio: {np.round(evr[:10], 4).tolist()}")
    for d in DIMS:
        print(f"[whiten]   first {d:3d} PCs explain {evr[:d].sum():.4f} of variance")

    base_val = baseline_ranking(Xdb, db_labels, Xva, yva)
    base_test = baseline_ranking(Xdb, db_labels, Xte, yte)

    configs = [("center", None)] + [("pca", d) for d in DIMS] + \
              [("pcaw", d) for d in DIMS] + [("zca", None)]

    val_rows = []
    for family, d in configs:
        t = make_transform(family, d, mu, lam, V)
        m = score(t, Xdb, db_labels, Xva, yva)
        label = family if d is None else f"{family}-{d}"
        val_rows.append((label, family, d, m))

    print_delta_table("VAL sweep (hyperparameter selection)", base_val,
                      [(lab, m) for lab, _, _, m in val_rows], SELECT_METRIC)

    best_label, best_family, best_d, best_val_m = max(val_rows, key=lambda r: r[3][SELECT_METRIC])
    print(f"\n[whiten] best on val by {SELECT_METRIC}: {best_label} "
          f"({best_val_m[SELECT_METRIC]:.4f} vs baseline {base_val[SELECT_METRIC]:.4f})")

    t = make_transform(best_family, best_d, mu, lam, V)
    test_m = score(t, Xdb, db_labels, Xte, yte)

    print_delta_table(f"TEST (selected config: {best_label})", base_test,
                      [(best_label, test_m)], SELECT_METRIC)
    print(f"\n[whiten] paper reference (raw cosine): acc@3={BASELINE_REF['acc@3']} "
          f"mvacc@3={BASELINE_REF['mvacc@3']}")

    save_results(out_name("retrieval_tcga_ot_whitening"), {
        "select_metric": SELECT_METRIC, "eps": EPS, "dims": DIMS,
        "explained_variance_ratio_top10": evr[:10].tolist(),
        "baseline": {"val": base_val, "test": base_test},
        "val_sweep": [{"config": lab, "family": f, "dim": d, **m} for lab, f, d, m in val_rows],
        "selected": {"config": best_label, "family": best_family, "dim": best_d},
        "test": test_m,
    })
    print("[whiten] OK")


if __name__ == "__main__":
    sys.exit(main())
