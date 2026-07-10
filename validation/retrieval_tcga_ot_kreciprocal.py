"""Tier-1 experiment 2/3: k-reciprocal re-ranking (Zhong et al., CVPR 2017).

Nothing is trained. We retrieve with plain cosine, then re-rank the result list using
neighbourhood *consistency* instead of raw similarity.

The idea: raw cosine says "b is near a". That is a one-way claim and it is easy to satisfy by
accident -- hub embeddings are near everything. k-reciprocal asks the stronger question: "is b
among a's top-k neighbours AND is a among b's top-k neighbours?" Mutual agreement is much
harder to satisfy by chance, so it filters out hubs and false matches. Each item is then
described by the (weighted) set of its k-reciprocal neighbours, and the final distance blends
the Jaccard distance between those neighbour sets with the original cosine distance:

    final = (1 - lambda) * jaccard + lambda * original

lambda=1 recovers the plain cosine ranking exactly -- included in the sweep as a sanity check
that the implementation is faithful. lambda=0 uses neighbourhood structure only.

TRANSDUCTIVE CAVEAT (important, and a real limitation of this technique): the canonical
algorithm pools queries and database together, so a query's re-ranking depends on the *other
queries* in the batch. That is fine for a benchmark, but it means the method as implemented
assumes you have the whole query set at once -- it is not a drop-in for a live one-query-at-a-
time search service. Numbers here therefore represent the technique's ceiling.

Hyperparameters (k1, k2, lambda) are selected on VAL queries and reported once on TEST queries.
Jaccard distance depends only on (k1, k2), so lambda is swept for free on top of each.
Env: SELECT_METRIC (default acc@3).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys

import numpy as np

from common import save_results
from retrieval_common import (load_retrieval_data, eval_ranking, l2_normalize,
                              baseline_ranking, print_delta_table, BASELINE_REF)

SELECT_METRIC = os.environ.get("SELECT_METRIC", "acc@3")
K1_GRID = [10, 20, 30]
K2_GRID = [3, 6]
LAMBDA_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]



# Results are written per selection metric, so sweeps selected on a different metric
# (e.g. SELECT_METRIC=mvacc@3) do not overwrite the default acc@3 run.
def out_name(stem):
    tag = "" if SELECT_METRIC == "acc@3" else "_sel-" + SELECT_METRIC.replace("@", "")
    return f"{stem}{tag}.json"

def precompute(Xq, Xdb, top_m):
    """Pooled [queries; database] squared-euclidean distances on L2-normalized vectors
    (monotonically equivalent to cosine), normalized as in the reference implementation.

    Returns (original_dist, initial_rank_topm, query_num). Only the top-M neighbours of each
    item are ever needed, so we partial-sort instead of fully ranking a ~10k x 10k matrix.
    """
    feats = np.concatenate([l2_normalize(Xq), l2_normalize(Xdb)], axis=0).astype(np.float32)
    query_num = Xq.shape[0]

    # ||a-b||^2 = 2 - 2 a.b for unit vectors
    original_dist = 2.0 - 2.0 * (feats @ feats.T)
    np.maximum(original_dist, 0, out=original_dist)
    # reference impl: normalize each column by its max, then transpose
    original_dist = (original_dist / original_dist.max(axis=0, keepdims=True)).T
    original_dist = np.ascontiguousarray(original_dist, dtype=np.float32)

    part = np.argpartition(original_dist, top_m, axis=1)[:, :top_m]
    rows = np.arange(original_dist.shape[0])[:, None]
    order = np.argsort(original_dist[rows, part], axis=1)
    initial_rank = part[rows, order].astype(np.int32)
    return original_dist, initial_rank, query_num


def k_reciprocal_neighbors(initial_rank, i, k):
    """Items that are in i's top-k AND have i in their own top-k."""
    fwd = initial_rank[i, : k + 1]
    bwd = initial_rank[fwd, : k + 1]
    return fwd[np.where(bwd == i)[0]]


def build_V(original_dist, initial_rank, k1):
    """Weighted neighbour-set descriptor V[i] over the pooled index space (pre-k2-smoothing).

    Depends only on k1, so callers cache it and apply smooth_V for each k2.
    """
    all_num = original_dist.shape[0]
    V = np.zeros((all_num, all_num), dtype=np.float16)

    for i in range(all_num):
        k_recip = k_reciprocal_neighbors(initial_rank, i, k1)
        expansion = k_recip
        # local query expansion: absorb a neighbour's own neighbourhood when the two
        # neighbourhoods substantially overlap (the 2/3 rule from the paper)
        for cand in k_recip:
            cand_recip = k_reciprocal_neighbors(initial_rank, cand, int(round(k1 / 2)))
            if len(np.intersect1d(cand_recip, k_recip)) > (2.0 / 3.0) * len(cand_recip):
                expansion = np.append(expansion, cand_recip)
        expansion = np.unique(expansion)
        weight = np.exp(-original_dist[i, expansion])
        V[i, expansion] = (weight / weight.sum()).astype(np.float16)
    return V


def smooth_V(V, initial_rank, k2):
    """Average each descriptor over its own k2 nearest neighbours. Accumulated row-block by
    row-block: the naive V[initial_rank[:, :k2], :] materializes an (N, k2, N) intermediate."""
    if k2 == 1:
        return V
    out = np.zeros(V.shape, dtype=np.float32)
    for j in range(k2):
        out += V[initial_rank[:, j], :]
    out /= k2
    return out.astype(np.float16)


def jaccard_distance(V, query_num):
    """1 - |min-intersection| / |max-union|, computed sparsely via an inverted index."""
    all_num = V.shape[0]
    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard = np.zeros((query_num, all_num), dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(all_num, dtype=np.float32)
        nz = np.where(V[i, :] != 0)[0]
        for j in nz:
            imgs = inv_index[j]
            temp_min[imgs] += np.minimum(V[i, j], V[imgs, j]).astype(np.float32)
        jaccard[i] = 1.0 - temp_min / (2.0 - temp_min)
    return jaccard


def main():
    sys.stdout.reconfigure(line_buffering=True)
    Xdb, db_labels, Xva, yva, Xte, yte = load_retrieval_data()

    base_val = baseline_ranking(Xdb, db_labels, Xva, yva)
    base_test = baseline_ranking(Xdb, db_labels, Xte, yte)
    top_m = max(K1_GRID) + 1

    def sweep(Xq, q_labels, grid):
        """grid: list of (k1, k2, lambda). V is cached per k1, jaccard per (k1, k2),
        and lambda only recombines two precomputed matrices -- so it is nearly free."""
        original_dist, initial_rank, qn = precompute(Xq, Xdb, top_m)
        rows, v_cache, jac_cache = [], {}, {}
        for k1, k2, lam in grid:
            if (k1, k2) not in jac_cache:
                if k1 not in v_cache:
                    print(f"  [k-recip] building V for k1={k1} ...", flush=True)
                    v_cache[k1] = build_V(original_dist, initial_rank, k1)
                print(f"  [k-recip] building jaccard for k1={k1} k2={k2} ...", flush=True)
                jac_cache[(k1, k2)] = jaccard_distance(
                    smooth_V(v_cache[k1], initial_rank, k2), qn)
            jac = jac_cache[(k1, k2)]
            final = jac * (1 - lam) + original_dist[:qn] * lam
            sim = -final[:, qn:]                     # gallery columns only; lower dist = higher sim
            m = eval_ranking(sim, db_labels, q_labels)
            rows.append((f"k1={k1} k2={k2} λ={lam}", (k1, k2, lam), m))
        return rows

    print(f"\n[k-recip] VAL sweep: {len(K1_GRID)*len(K2_GRID)} jaccard builds "
          f"x {len(LAMBDA_GRID)} lambdas (lambda is free)")
    val_grid = [(k1, k2, lam) for k1 in K1_GRID for k2 in K2_GRID for lam in LAMBDA_GRID]
    val_rows = sweep(Xva, yva, val_grid)

    print_delta_table("VAL sweep (hyperparameter selection)", base_val,
                      [(lab, m) for lab, _, m in val_rows], SELECT_METRIC)

    # sanity: lambda=1.0 must reproduce the plain cosine ranking
    lam1 = [m for lab, (k1, k2, lam), m in val_rows if lam == 1.0][0]
    ok = abs(lam1[SELECT_METRIC] - base_val[SELECT_METRIC]) < 1e-9
    print(f"\n[k-recip] sanity check -- λ=1.0 reproduces raw cosine on val: "
          f"{'PASS' if ok else 'FAIL'} ({lam1[SELECT_METRIC]:.4f} vs {base_val[SELECT_METRIC]:.4f})")

    best_label, best_params, best_val_m = max(val_rows, key=lambda r: r[2][SELECT_METRIC])
    print(f"[k-recip] best on val by {SELECT_METRIC}: {best_label} "
          f"({best_val_m[SELECT_METRIC]:.4f} vs baseline {base_val[SELECT_METRIC]:.4f})")

    print(f"\n[k-recip] evaluating selected config on TEST ...")
    test_rows = sweep(Xte, yte, [best_params])
    test_m = test_rows[0][2]

    print_delta_table(f"TEST (selected config: {best_label})", base_test,
                      [(best_label, test_m)], SELECT_METRIC)
    print(f"\n[k-recip] paper reference (raw cosine): acc@3={BASELINE_REF['acc@3']} "
          f"mvacc@3={BASELINE_REF['mvacc@3']}")

    save_results(out_name("retrieval_tcga_ot_kreciprocal"), {
        "select_metric": SELECT_METRIC, "k1_grid": K1_GRID, "k2_grid": K2_GRID,
        "lambda_grid": LAMBDA_GRID, "transductive": True,
        "lambda1_sanity_pass": bool(ok),
        "baseline": {"val": base_val, "test": base_test},
        "val_sweep": [{"config": lab, "k1": p[0], "k2": p[1], "lambda": p[2], **m}
                      for lab, p, m in val_rows],
        "selected": {"config": best_label, "k1": best_params[0], "k2": best_params[1],
                     "lambda": best_params[2]},
        "test": test_m,
    })
    print("[k-recip] OK")


if __name__ == "__main__":
    sys.exit(main())
