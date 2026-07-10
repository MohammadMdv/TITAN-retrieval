"""Tier-1 experiment 3/3: alpha-weighted query expansion (αQE) and database-side augmentation (DBA).

Nothing is trained. Both techniques replace a vector with a similarity-weighted average of
itself and its nearest neighbours, then re-normalize. The bet is that averaging over a small
neighbourhood cancels slide-specific noise (staining, scanner, tissue-cut artifacts) while
reinforcing the shared morphological signal -- a denoising step, not a learned one.

  αQE : expand the QUERY using its top-k neighbours in the database, then search again.
        Costs one extra search per query. Fully inductive -- each query is expanded using
        only the database, so nothing leaks between queries and this works one-query-at-a-time
        in a live system.

  DBA : expand every DATABASE vector using its own top-k database neighbours, once, offline.
        Queries are left untouched. Also fully inductive, and free at query time.

Both weight neighbour i by sim_i^alpha (alpha=0 -> plain mean; larger alpha -> trust the
closest neighbours more). Similarities are clamped at 0 so a negative cosine can never flip a
neighbour's sign under an even alpha.

Self-exclusion matters: a database vector is its own nearest neighbour, so DBA's neighbour
search must skip the diagonal or it degenerates toward the identity. Handled explicitly below.

QE and DBA are reported SEPARATELY (that is the point of this script). Their combination is
also shown, clearly labelled, since the two are the same family and compose naturally.

Hyperparameters (k, alpha) are selected on VAL queries and reported once on TEST queries.
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
K_GRID = [1, 2, 3, 5, 10]
ALPHA_GRID = [0.0, 1.0, 2.0, 3.0]



# Results are written per selection metric, so sweeps selected on a different metric
# (e.g. SELECT_METRIC=mvacc@3) do not overwrite the default acc@3 run.
def out_name(stem):
    tag = "" if SELECT_METRIC == "acc@3" else "_sel-" + SELECT_METRIC.replace("@", "")
    return f"{stem}{tag}.json"

def _expand(Xsrc_n, Xref_n, k, alpha, exclude_self):
    """Weighted-average each row of Xsrc_n over its top-k neighbours in Xref_n (+ itself).

    Both inputs must already be L2-normalized. Returns a re-normalized matrix.
    """
    sim = Xsrc_n @ Xref_n.T
    if exclude_self:
        np.fill_diagonal(sim, -np.inf)  # a db vector must not retrieve itself as a neighbour

    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    nn = nn[rows, np.argsort(-sim[rows, nn], axis=1)]        # top-k, sorted by similarity

    w = np.clip(sim[rows, nn], 0.0, None) ** alpha           # (n, k) neighbour weights
    neigh = (w[:, :, None] * Xref_n[nn]).sum(axis=1)         # weighted neighbour sum
    return l2_normalize(Xsrc_n + neigh)                      # the vector itself carries weight 1


def apply_qe(Xdb_n, Xq_n, k, alpha):
    """Expand queries against the (possibly already DBA'd) database."""
    return _expand(Xq_n, Xdb_n, k, alpha, exclude_self=False)


def apply_dba(Xdb_n, k, alpha):
    """Expand each database vector against the rest of the database."""
    return _expand(Xdb_n, Xdb_n, k, alpha, exclude_self=True)


def score_qe(Xdb_n, db_labels, Xq_n, q_labels, k, alpha):
    Zq = apply_qe(Xdb_n, Xq_n, k, alpha)
    return eval_ranking(Zq @ Xdb_n.T, db_labels, q_labels)


def score_dba(Xdb_n, db_labels, Xq_n, q_labels, k, alpha):
    Zdb = apply_dba(Xdb_n, k, alpha)
    return eval_ranking(Xq_n @ Zdb.T, db_labels, q_labels)


def score_both(Xdb_n, db_labels, Xq_n, q_labels, kd, ad, kq, aq):
    Zdb = apply_dba(Xdb_n, kd, ad)
    Zq = apply_qe(Zdb, Xq_n, kq, aq)   # queries expanded against the augmented database
    return eval_ranking(Zq @ Zdb.T, db_labels, q_labels)


def sweep(score_fn, Xdb_n, db_labels, Xq_n, q_labels):
    rows = []
    for k in K_GRID:
        for a in ALPHA_GRID:
            m = score_fn(Xdb_n, db_labels, Xq_n, q_labels, k, a)
            rows.append((f"k={k} α={a}", (k, a), m))
    return rows


def report(name, rows, base_val, base_test, Xdb_n, db_labels, Xte_n, yte, score_fn):
    print_delta_table(f"{name}: VAL sweep (hyperparameter selection)", base_val,
                      [(lab, m) for lab, _, m in rows], SELECT_METRIC)
    best_label, best_params, best_val_m = max(rows, key=lambda r: r[2][SELECT_METRIC])
    beat = best_val_m[SELECT_METRIC] > base_val[SELECT_METRIC]
    print(f"\n[{name}] best on val by {SELECT_METRIC}: {best_label} "
          f"({best_val_m[SELECT_METRIC]:.4f} vs baseline {base_val[SELECT_METRIC]:.4f})"
          f"{'' if beat else '  -- does NOT beat baseline on val'}")

    test_m = score_fn(Xdb_n, db_labels, Xte_n, yte, *best_params)
    print_delta_table(f"{name}: TEST (selected config: {best_label})", base_test,
                      [(best_label, test_m)], SELECT_METRIC)
    return {"selected": {"config": best_label, "k": best_params[0], "alpha": best_params[1]},
            "val_sweep": [{"config": lab, "k": p[0], "alpha": p[1], **m} for lab, p, m in rows],
            "val_beats_baseline": bool(beat), "test": test_m}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    Xdb, db_labels, Xva, yva, Xte, yte = load_retrieval_data()
    Xdb_n, Xva_n, Xte_n = l2_normalize(Xdb), l2_normalize(Xva), l2_normalize(Xte)

    base_val = baseline_ranking(Xdb, db_labels, Xva, yva)
    base_test = baseline_ranking(Xdb, db_labels, Xte, yte)
    results = {"select_metric": SELECT_METRIC, "k_grid": K_GRID, "alpha_grid": ALPHA_GRID,
               "baseline": {"val": base_val, "test": base_test}}

    print("\n[qe] === alpha-weighted QUERY EXPANSION (queries expanded, database untouched) ===")
    qe_rows = sweep(score_qe, Xdb_n, db_labels, Xva_n, yva)
    results["qe"] = report("qe", qe_rows, base_val, base_test, Xdb_n, db_labels, Xte_n, yte, score_qe)

    print("\n\n[dba] === DATABASE-SIDE AUGMENTATION (database expanded, queries untouched) ===")
    dba_rows = sweep(score_dba, Xdb_n, db_labels, Xva_n, yva)
    results["dba"] = report("dba", dba_rows, base_val, base_test, Xdb_n, db_labels, Xte_n, yte, score_dba)

    # Combination, using each technique's own val-selected hyperparameters. Labelled as a
    # bonus: the brief was to measure QE and DBA in isolation first.
    kd, ad = results["dba"]["selected"]["k"], results["dba"]["selected"]["alpha"]
    kq, aq = results["qe"]["selected"]["k"], results["qe"]["selected"]["alpha"]
    print("\n\n[qe+dba] === COMBINED (bonus; each technique keeps its own val-selected config) ===")
    comb_val = score_both(Xdb_n, db_labels, Xva_n, yva, kd, ad, kq, aq)
    comb_test = score_both(Xdb_n, db_labels, Xte_n, yte, kd, ad, kq, aq)
    label = f"dba(k={kd},α={ad}) + qe(k={kq},α={aq})"
    print_delta_table("COMBINED: VAL", base_val, [(label, comb_val)], SELECT_METRIC)
    print_delta_table("COMBINED: TEST", base_test, [(label, comb_test)], SELECT_METRIC)
    results["qe_plus_dba"] = {"config": label, "val": comb_val, "test": comb_test}

    print(f"\n[qe/dba] paper reference (raw cosine): acc@3={BASELINE_REF['acc@3']} "
          f"mvacc@3={BASELINE_REF['mvacc@3']}")
    save_results(out_name("retrieval_tcga_ot_query_expansion"), results)
    print("[qe/dba] OK")


if __name__ == "__main__":
    sys.exit(main())
