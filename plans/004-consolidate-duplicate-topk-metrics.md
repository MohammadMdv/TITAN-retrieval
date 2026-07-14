# Plan 004 — Three divergent copies of the top-k retrieval metric; the Tier-1 deltas compare across two of them

**Written against commit:** `9d56a69` · **re-stamped and drift-checked at `b803ba5`** (post-001/002/003 merge)
**Category:** tech debt / correctness hazard
**Impact:** MEDIUM · **Effort:** S · **Risk of fix:** MEDIUM (touches the code that produces published numbers — the guardrails in §5 are not optional)
**Depends on:** plan 003 — **DONE and merged.** Its `test_metric_implementations_agree.py` is the
safety net that makes this refactor provable rather than hopeful.

> **Drift check at `b803ba5`.** All four files this plan edits — `retrieval_tcga_ot.py`,
> `subtasks_tcga_ot.py`, `retrieval_common.py`, `retrieval_tcga_ot_kreciprocal.py` — are
> **untouched** by plans 001/002/003, so every excerpt below is still accurate.
>
> Two things changed around them:
> 1. **`validation/tests/` now holds 47 passing tests**, including
>    `test_metric_implementations_agree.py`, which pins the three implementations to each other.
>    That is this plan's safety net; `pytest` is installed and the suite runs offline on CPU in
>    ~4s. Run it before and after.
> 2. **`save_results` now stamps a `_provenance` key** (plan 002). So re-running the scripts in
>    §4b will add `_provenance` to the regenerated JSONs. That is expected — `make_results.py`
>    already skips `_`-prefixed keys, and §4b's assertion script looks up task names explicitly,
>    so it is unaffected. Do not strip the stamp.
>
> **`TCGA_TITAN_features.pkl` is confirmed present in the local HF cache** and loads under
> `HF_HUB_OFFLINE=1` (verified: db=8226, val_q=1612, test_q=1348). §4b therefore runs offline.

---

## 1. Why this matters

The single most safety-critical function in this repo — "given a similarity matrix, compute
Acc@k / MVAcc@k" — is implemented **three times**, independently:

**Copy 1** — `validation/retrieval_common.py:71`, used by BRACS and every Tier-1 experiment:

```python
def eval_ranking(sim, db_labels, q_labels, k=K):
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
```

**Copy 2** — `validation/retrieval_tcga_ot.py:41`, used by the **headline TCGA-OT baseline**
(the 0.8717 / 0.7812 that `RESULTS.md` reports against the paper's 0.880 / 0.807):

```python
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
```

**Copy 3** — `validation/subtasks_tcga_ot.py:73`, used by all five sub-task rows, and it uses a
*different mechanism* for the tie-break:

```python
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
        mv3 += Counter(retr).most_common(1)[0][0] == q_lab[i]      # <-- different tie-break
    return {"acc@1": a1 / len(q_lab), "mvacc@3": mv3 / len(q_lab)}
```

**They agree today.** I checked: `Counter.most_common(1)` sorts by count with a *stable* sort, so
ties keep insertion order, and `retr` is in descending-similarity order — which is exactly what
copies 1 and 2 achieve explicitly via `cands[0]`. Copy 3 gets the right answer *by accident of a
CPython implementation detail that is not part of `Counter`'s documented contract.*

The live hazard is the coupling this creates. `retrieval_common.py:23` hardcodes:

```python
# Raw-cosine baseline on this exact protocol (see retrieval_tcga_ot.py); paper refs 0.880/0.807.
BASELINE_REF = {"acc@3": 0.8717, "mvacc@3": 0.7812}
```

Those two numbers were produced by **copy 2**. Every Tier-1 delta in `RESULTS.md` — the whole
"none of them beat the baseline" table — is computed with **copy 1**. The comparison is only valid
because the two copies happen to agree. If anyone ever "improves" one of them (a faster
`argpartition`, a different tie rule, a `stable=True` sort), the Tier-1 deltas silently become an
apples-to-oranges comparison and **nothing in the repo would catch it**.

Also in scope, because it is the same class of defect — a correctness check that does not check:

```python
# validation/retrieval_tcga_ot_kreciprocal.py:170-173
    lam1 = [m for lab, (k1, k2, lam), m in val_rows if lam == 1.0][0]
    ok = abs(lam1[SELECT_METRIC] - base_val[SELECT_METRIC]) < 1e-9
    print(f"\n[k-recip] sanity check -- λ=1.0 reproduces raw cosine on val: "
          f"{'PASS' if ok else 'FAIL'} ({lam1[SELECT_METRIC]:.4f} vs {base_val[SELECT_METRIC]:.4f})")
```

On `FAIL` this prints the word "FAIL" and then carries on to save the results and report a
scientific conclusion drawn from a re-ranker that just failed its own identity test.

## 2. The fix

One implementation (`retrieval_common.eval_ranking`), called from all three sites. The
`argsort`-based ranking loop and its tie-break stay **byte-for-byte as they are in copy 1** — this
is a de-duplication, not an optimization. Then make the λ=1.0 check raise.

## 2b. ⚠️ PLAN REVISION (after a first execution attempt stopped) — read before §3

The first executor **correctly refused to execute this plan**, and found two defects in it. Both
are mine. The steps below are revised accordingly; §3's Step 0 now begins with a mandatory
**Phase A** that must land *before* any production code is touched.

**Defect 1 — the plan contradicted itself.** Step 2 says delete `topk_metrics`; §6 says
`validation/tests/**` is out of scope. But plan 003's `test_metric_implementations_agree.py` does
`from retrieval_tcga_ot import topk_metrics` at module scope. Deleting the function makes that test
file un-importable, so "delete `topk_metrics`" and "47 tests still pass, don't edit tests" cannot
both hold. **`validation/tests/test_metric_implementations_agree.py` is now IN SCOPE.**

More fundamentally: an "these three implementations agree" test is *self-destroying* — after
consolidation there are no longer three implementations to compare. The guarantee has to be
carried across the refactor differently: **pin the outputs, not the implementations.**

**Defect 2 — plan 003's test fixture is vacuous, so this plan's safety net is an illusion.**
Reviewer-verified at `b803ba5`: `_synthetic` builds class centres at `rng.normal(...) * 3` with
unit noise, which separates the classes so completely that **every metric returns exactly 1.0 on
all 20 seeds**. So `test_eval_ranking_matches_topk_metrics` and
`test_agreement_holds_across_many_random_problems` are asserting `1.0 == 1.0`. They would pass with
a badly broken implementation. The fixture's own comment — "class-structured embeddings, so the
metrics are non-trivial (not all 0, not all 1)" — is **false**.

Fixed by changing the centre separation from `* 3` to `* 0.45`. Verified at that setting:
metrics land at acc@1≈0.56 / acc@3≈0.84 / mvacc@3≈0.58, **no seed saturates**, `mvacc@3 ≠ acc@3`
(so majority-vote is exercised as distinct from acc@k), and the **MV tie-break branch actually
fires on 18% of queries (90/500)** instead of never. All three implementations still agree on this
harder fixture — confirmed on all 20 seeds.

## 3. Steps

### Step 0 — precondition

```bash
cd /home/user01/TITAN && .venv/bin/python -m pytest validation/tests -q
```

Must be **green** before you touch anything. If plan 003 has not landed, stop here.

### Step 0.5 — PHASE A: repair the safety net FIRST (before any production edit)

Everything in this step happens with `topk_metrics` **still present**, so it proves the three
implementations agree on a fixture that can actually tell them apart, and freezes that agreement
into golden values that survive the deletion.

**(a) De-vacuum the fixture.** In `validation/tests/test_metric_implementations_agree.py`, change
`_synthetic`'s centre separation and fix the lying comment:

```python
def _synthetic(seed=0, n_db=60, n_q=25, dim=16, n_cls=4):
    rng = np.random.default_rng(seed)
    labels = np.array([f"C{i % n_cls}" for i in range(n_db)])
    q_labels = np.array([f"C{i % n_cls}" for i in range(n_q)])
    # Centres at *0.45, NOT *3. At *3 the classes separate so cleanly that every metric returns
    # exactly 1.0 on every seed, and the agreement tests below degenerate into `1.0 == 1.0` --
    # they would pass against a badly broken implementation. At *0.45 the metrics land around
    # acc@1 0.56 / acc@3 0.84 / mvacc@3 0.58, no seed saturates, mvacc@3 differs from acc@3, and
    # the majority-vote TIE-BREAK branch is exercised on ~18% of queries. Do not raise this.
    centers = rng.normal(size=(n_cls, dim)) * 0.45
    Xdb = np.stack([centers[int(l[1:])] + rng.normal(size=dim) for l in labels])
    Xq = np.stack([centers[int(l[1:])] + rng.normal(size=dim) for l in q_labels])
    return Xdb.astype(np.float32), labels, Xq.astype(np.float32), q_labels
```

**(b) Add a guard so the fixture can never silently go vacuous again**, and the golden table.
Append to the same file:

```python
# (acc@1, acc@3, mvacc@3) produced by the THREE pre-consolidation implementations, which were
# verified to agree with each other on every one of these seeds at commit b803ba5. These literals
# are what carries the guarantee ACROSS the consolidation: after topk_metrics is deleted there is
# no second implementation left to compare against, so the single survivor must instead reproduce
# exactly what the three of them agreed on. Captured by the reviewer from the pre-refactor code.
GOLDEN = {
    0: (0.36, 0.72, 0.44),   1: (0.48, 0.72, 0.48),   2: (0.36, 0.72, 0.48),
    3: (0.60, 0.84, 0.68),   4: (0.60, 0.92, 0.60),   5: (0.56, 0.92, 0.64),
    6: (0.64, 0.84, 0.68),   7: (0.68, 0.88, 0.64),   8: (0.68, 0.84, 0.64),
    9: (0.52, 0.76, 0.56),  10: (0.60, 0.72, 0.40),  11: (0.44, 0.76, 0.48),
    12: (0.52, 0.92, 0.44), 13: (0.56, 0.92, 0.48),  14: (0.68, 0.84, 0.68),
    15: (0.64, 0.96, 0.68), 16: (0.48, 0.84, 0.52),  17: (0.64, 0.96, 0.64),
    18: (0.64, 0.92, 0.72), 19: (0.56, 0.88, 0.68),
}


def test_the_fixture_is_actually_discriminative():
    """A saturated fixture makes every agreement test below vacuous. This is the tripwire.

    The original fixture (centres at *3) returned 1.0 for every metric on every seed -- so the
    agreement assertions were comparing 1.0 == 1.0 and proved nothing. Never let that recur.
    """
    accs = [eval_ranking(cosine_sim(*_synthetic(seed=s)[2::-2][::-1]  # noqa: E501 - see below
                                    ), None, None) for s in []]  # placeholder, replaced below
```

> ⚠️ Do **not** copy that last stub — write `test_the_fixture_is_actually_discriminative` yourself,
> straightforwardly, so that it asserts, across seeds 0..19: every `acc@3` is `< 1.0` for at least
> 15 of the 20 seeds; `mvacc@3 != acc@3` on at least 10 seeds; and the mean `acc@1` is strictly
> between 0.3 and 0.8. Keep it simple and readable. Its job is to fail loudly if anyone ever makes
> the fixture easy again.

**(c) Add the golden test** (this is the one that survives Phase B):

```python
def test_eval_ranking_reproduces_the_pre_consolidation_golden_values():
    """The single surviving implementation must reproduce, exactly, what the three agreed on."""
    for seed, want in GOLDEN.items():
        Xdb, db_l, Xq, q_l = _synthetic(seed=seed)
        m = eval_ranking(cosine_sim(Xq, Xdb), db_l, q_l, k=3)
        got = (m["acc@1"], m["acc@3"], m["mvacc@3"])
        assert got == pytest.approx(want, abs=1e-12), (seed, got, want)
```

**(d) Run the suite NOW, with `topk_metrics` still present.** It must be green. That green run is
the evidence that the three implementations agree on a fixture that can actually distinguish them,
and that `GOLDEN` faithfully records what they agree on. **If it is not green, STOP and report** —
the golden values or the fixture are wrong, and nothing further may proceed.

### Step 1 — record the numbers you must not change

```bash
cd /home/user01/TITAN
.venv/bin/python -c "
import json
for f in ['retrieval_tcga_ot', 'subtasks_tcga_ot']:
    print(f, json.load(open(f'validation/results/{f}.json')))
" > /tmp/before_004.txt
cat /tmp/before_004.txt
```

The values that must survive this refactor, from `validation/results/`:

| Source | Metric | Value |
|---|---|---|
| `retrieval_tcga_ot.json` | `acc@3` | **0.8717** |
| `retrieval_tcga_ot.json` | `mvacc@3` | **0.7812** |
| `subtasks_tcga_ot.json` | `NSCLC_LUAD_vs_LUSC` acc@1 / mvacc@3 | **0.8993 / 0.9065** |
| `subtasks_tcga_ot.json` | `RCC_3way` acc@1 / mvacc@3 | **0.8830 / 0.9149** |
| `subtasks_tcga_ot.json` | `BRCA_IDC_vs_ILC` acc@1 / mvacc@3 | **0.9254 / 0.9403** |
| `subtasks_tcga_ot.json` | `Sarcoma_4way` acc@1 / mvacc@3 | **0.4074 / 0.3704** |
| `subtasks_tcga_ot.json` | `Brain_GBM_vs_LGG` acc@1 / mvacc@3 | **0.8605 / 0.8605** |

### Step 2 — `retrieval_tcga_ot.py` uses the shared metric

Delete `topk_metrics` (lines 41–56) and the now-unused `from collections import Counter`.
Import the shared helpers, and note `retrieval_common` already has `load_embeddings` and
`build_split` — but they are **not identical** to this file's local copies (`build_split` here
takes `target` as a positional argument and also returns `slide_id`s). **Do not** try to unify
those in this plan; only the metric moves. Change the import block to add:

```python
from retrieval_common import eval_ranking
```

and replace the call site:

```python
    # was: acc3, mv3 = topk_metrics(sim, db_labels, q_labels, K=K)
    m = eval_ranking(sim, db_labels, q_labels, k=K)
    acc3, mv3 = m["acc@3"], m["mvacc@3"]
    print(f"[ret-tcga-ot] Acc@{K}={acc3:.4f} (ref 0.880)  MVAcc@{K}={mv3:.4f} (ref 0.807)")
```

`eval_ranking` also returns `acc@1`, which `topk_metrics` did not. Add it to the saved JSON — it
is free information and `make_results.py` only reads `acc@3` / `mvacc@3`, so nothing downstream
breaks:

```python
    save_results("retrieval_tcga_ot.json", {
        "K": K, "db_size": len(db_ids), "n_queries": len(q_ids), "n_leaking_dropped": n_drop,
        "acc@1": m["acc@1"], "acc@3": acc3, "mvacc@3": mv3,
        "reference": {"acc@3": 0.880, "mvacc@3": 0.807},
        "patient_disjoint_asserted": True,
    })
```

### Step 3 — `subtasks_tcga_ot.py` uses the shared metric

Replace the body of `retrieval()` (lines 73–86). **Keep the patient-disjointness assertion and
keep the function's return shape** (`{"acc@1", "mvacc@3"}`) — `make_results.py` indexes
`r['retrieval']['acc@1']` and `r['retrieval']['mvacc@3']`.

```python
def retrieval(tr, te, label_col="_y"):
    from retrieval_common import cosine_sim, eval_ranking
    Xdb = np.stack(tr["emb"].values).astype(np.float32)
    Xq = np.stack(te["emb"].values).astype(np.float32)
    db_lab, q_lab = tr[label_col].values, te[label_col].values
    assert set(tr["case_id"]).isdisjoint(te["case_id"]), "patient leak in retrieval"
    m = eval_ranking(cosine_sim(Xq, Xdb), db_lab, q_lab, k=3)
    return {"acc@1": m["acc@1"], "mvacc@3": m["mvacc@3"]}
```

Note `cosine_sim` L2-normalizes with the same `+1e-8` guard (`retrieval_common.l2_normalize`), so
this is numerically the same operation the old inline code did. Drop the now-unused
`from collections import Counter` **only if** nothing else in the file uses it — `run_task` uses
`Counter(te["_y"])` at line 98, so **keep the import**. Read before deleting.

### Step 3.5 — PHASE B: retire the two now-impossible tests

Once `topk_metrics` is gone, these two tests in `test_metric_implementations_agree.py` cannot even
import, let alone run:

- `test_eval_ranking_matches_topk_metrics`
- `test_agreement_holds_across_many_random_problems`

**Delete both, and delete the `from retrieval_tcga_ot import topk_metrics` line.** Their guarantee
is not lost — it now lives in `test_eval_ranking_reproduces_the_pre_consolidation_golden_values`
(Step 0.5c), which pins the survivor to exactly what all three agreed on, and which keeps working
forever precisely *because* it depends on no second implementation.

**Keep** `test_eval_ranking_matches_subtasks_retrieval` and
`test_subtasks_retrieval_asserts_patient_disjointness` — `subtasks_tcga_ot.retrieval` still exists
as a function after consolidation (it now wraps `eval_ranking`), so both still run and both still
pin a real end-to-end path. Also keep `test_the_fixture_is_actually_discriminative`.

Update the module docstring: it currently describes three implementations in the present tense.
Rewrite it to say there is now **one**, that `GOLDEN` records what the three agreed on before
consolidation at `b803ba5`, and that the golden test is what carries that guarantee forward.

Net test count: −2 (retired) +2 (fixture tripwire, golden) = **47**. Do not chase the number; the
criterion is **0 failed**.

### Step 4 — make `BASELINE_REF` honest about where it came from

In `validation/retrieval_common.py`, the hardcoded baseline is fine to keep (the Tier-1 scripts
also recompute the baseline themselves via `baseline_ranking`, so `BASELINE_REF` is only used for
the "paper reference" print). Just make its provenance explicit so nobody treats it as ground
truth:

```python
# Raw-cosine baseline on this exact protocol, as measured by retrieval_tcga_ot.py at commit
# 9d56a69 and recorded in results/retrieval_tcga_ot.json. Kept here only for the reference line
# the Tier-1 scripts print; every Tier-1 DELTA is computed against a freshly recomputed
# baseline_ranking(), never against these constants. Paper refs: 0.880 / 0.807.
BASELINE_REF = {"acc@3": 0.8717, "mvacc@3": 0.7812}
```

Verify that claim before writing the comment: `grep -n 'BASELINE_REF' validation/*.py`. It should
appear only in `retrieval_common.py` (definition) and in the three Tier-1 scripts' final `print`
of the paper reference. If it is used in a *delta* computation anywhere, stop — that is a bigger
finding than this plan covers, and it needs reporting.

### Step 5 — make the k-reciprocal λ=1.0 check fail the run

In `validation/retrieval_tcga_ot_kreciprocal.py`, after the existing `print` of the sanity check
(line 173), add:

```python
    if not ok:
        sys.exit("[k-recip] ABORT: λ=1.0 must reproduce the raw-cosine ranking exactly. It does "
                 "not, so the re-ranking implementation is wrong and any conclusion drawn from "
                 "this run would be meaningless. Not saving results.")
```

Placed **before** `save_results`, so a broken run cannot write a result file. `sys` is already
imported at line 31.

## 4. Verification

### 4a — unit (fast, no network/GPU)

```bash
cd /home/user01/TITAN && .venv/bin/python -m pytest validation/tests -q
```

`test_metric_implementations_agree.py` from plan 003 now exercises the *consolidated* code paths
through the same three public entry points. It must still be green. **This is the whole point of
the dependency on plan 003** — the test was written against the three separate implementations,
so it passing afterwards proves the consolidation preserved behavior.

### 4b — the real numbers must not move (needs the HF-cached pkl; no GPU)

`TCGA_TITAN_features.pkl` is already in the local HF cache, so this runs offline:

```bash
cd /home/user01/TITAN/validation
HF_HUB_OFFLINE=1 ../.venv/bin/python retrieval_tcga_ot.py
HF_HUB_OFFLINE=1 N_JOBS=10 ../.venv/bin/python subtasks_tcga_ot.py
```

Then assert the numbers against the table in Step 1:

```bash
cd /home/user01/TITAN && .venv/bin/python -c "
import json
r = json.load(open('validation/results/retrieval_tcga_ot.json'))
assert abs(r['acc@3']   - 0.8717) < 5e-5, r['acc@3']
assert abs(r['mvacc@3'] - 0.7812) < 5e-5, r['mvacc@3']
s = json.load(open('validation/results/subtasks_tcga_ot.json'))
want = {'NSCLC_LUAD_vs_LUSC': (0.8993, 0.9065), 'RCC_3way': (0.8830, 0.9149),
        'BRCA_IDC_vs_ILC': (0.9254, 0.9403), 'Sarcoma_4way': (0.4074, 0.3704),
        'Brain_GBM_vs_LGG': (0.8605, 0.8605)}
for t, (a1, mv) in want.items():
    got = s[t]['retrieval']
    assert abs(got['acc@1'] - a1) < 5e-4, (t, got['acc@1'], a1)
    assert abs(got['mvacc@3'] - mv) < 5e-4, (t, got['mvacc@3'], mv)
print('ALL PUBLISHED RETRIEVAL NUMBERS UNCHANGED')
"
```

> `subtasks_tcga_ot.py` also refits linear probes (a 45-point C sweep × 5 tasks). That part is
> untouched by this plan; the LP numbers should also be unchanged, but they are not what this
> check is guarding. It takes a few minutes.

### 4c — the λ=1.0 abort actually aborts

Do not run the full k-reciprocal sweep (it is slow). Prove the guard with an inline check:

```bash
cd /home/user01/TITAN && .venv/bin/python -c "
import ast, sys
src = open('validation/retrieval_tcga_ot_kreciprocal.py').read()
i_sanity = src.index('lambda1 sanity' if 'lambda1 sanity' in src else 'sanity check')
i_abort  = src.index('ABORT')
i_save   = src.index('save_results(')
assert i_sanity < i_abort < i_save, 'the abort must sit between the check and the save'
ast.parse(src)
print('ABORT GUARD IS IN THE RIGHT PLACE')
"
```

## 5. Done criteria (machine-checkable)

- [ ] `.venv/bin/python -m pytest validation/tests -q` → 0 failed.
- [ ] The §4b assertion script prints `ALL PUBLISHED RETRIEVAL NUMBERS UNCHANGED`.
- [ ] `grep -n 'def topk_metrics' validation/retrieval_tcga_ot.py` → no match (copy 2 is gone).
- [ ] `grep -n 'most_common' validation/subtasks_tcga_ot.py` → matches **only** inside `run_task`
      (the `Counter(te["_y"])` class-count line), never inside `retrieval`.
- [ ] `grep -rn 'def eval_ranking\|def topk_metrics' validation/*.py` → exactly **one** hit
      (`retrieval_common.py`).
- [ ] §4c prints `ABORT GUARD IS IN THE RIGHT PLACE`.
- [ ] `git diff --stat` touches only: `retrieval_tcga_ot.py`, `subtasks_tcga_ot.py`,
      `retrieval_common.py`, `retrieval_tcga_ot_kreciprocal.py`.

## 6. Scope

**In scope:** `validation/retrieval_tcga_ot.py`, `validation/subtasks_tcga_ot.py`,
`validation/retrieval_common.py` (comment only), `validation/retrieval_tcga_ot_kreciprocal.py`
(the abort guard only), and — **added in the §2b revision** —
`validation/tests/test_metric_implementations_agree.py` (Steps 0.5 and 3.5).

**Out of scope — do not touch:**
- **Every test file under `validation/tests/` EXCEPT `test_metric_implementations_agree.py`.**
  The other five are off-limits: run them, do not edit them.
- **The body of `eval_ranking` itself.** It is the survivor. Do not "improve" it while
  consolidating — no `argpartition`, no vectorization, no changed tie rule. A faster ranking loop
  is worth roughly nothing here (the matrices are ~1.3k × 8.2k) and risks everything. If you
  believe it is wrong, stop and report; do not fix it inside a refactor whose entire premise is
  that no number moves.
- `retrieval_bracs.py` and `finetune_bracs_lora.py` — they **already** call the shared
  `eval_ranking`. Nothing to do.
- `retrieval_camelyon16.py` — it has its own `evaluate()` with a genuinely different metric
  (Top-1 / Top-3-MV / Top-5-MV over a *different* protocol, for comparison against an external
  PathSearch run). It is **not** a fourth copy of the same thing. Leave it alone.
- The local `load_embeddings` / `build_split` duplication between `retrieval_tcga_ot.py` and
  `retrieval_common.py`. Real, but a separate (and lower-value) change — do not fold it in.
- Any `validation/results/*.json` — they are *regenerated* by §4b, never hand-edited.
- `validation/RESULTS.md` — no number in it should change. If one does, §7 applies.

## 7. Escape hatch — the one that matters

**If any published number in §4b moves, even in the fourth decimal: STOP. Revert. Report.**

The entire premise of this plan is that the three implementations are behaviorally identical and
the consolidation is a no-op on every output. If a number moves, that premise was false — which
means one of the three copies has a real bug and *has been producing a published number with it*.
That is a much more serious finding than the tech debt this plan set out to remove, and it must be
reported and adjudicated on its own, not buried inside a refactor commit.

Do not adjust a tolerance to make the check pass.

## 8. Other escape hatches — STOP and report back if:

- `BASELINE_REF` turns out to be used in an actual delta computation (Step 4), not just a print.
- `Counter` is needed in `subtasks_tcga_ot.retrieval` for something you did not anticipate.
- Plan 003's `test_metric_implementations_agree.py` does not exist. Do not proceed without it and
  do not write a hasty substitute — this refactor is only safe because that test is there.
