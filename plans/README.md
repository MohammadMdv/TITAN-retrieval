# Improvement plans — TITAN validation harness

Audit run: **2026-07-12**, against commit **`9d56a69`** (branch `main`).
Effort level: `standard` — hotspot-weighted, all nine categories.

This repo is a research harness: its product is ~30 numbers in `validation/RESULTS.md`. The audit
was therefore weighted toward **scientific correctness and reproducibility** (leakage, selection
bias, metric bugs, seeding) over conventional software concerns. That weighting is why a
"reporting script overwrites a markdown file" ranks above anything in the security category here:
in this repo, the markdown file *is* the deliverable.

**Selection was made non-interactively** (no user available to choose): the four plans below are
the top items by leverage. Nothing was written for the minor items in §5.

---

## Execution order

```
001 ──┐
      ├──> 002        (both edit make_results.py; 002 adds a `_provenance` key that
      │                 001's restructured subtasks loop must skip)
003 ──┴──> 004        (004 is a refactor of published-number code; 003's
                       test_metric_implementations_agree.py is its safety net)
```

`001` and `003` are independent and can go in parallel. `002` needs `001`. `004` needs `003`.

If you only do one thing: **001** — it is a one-file fix that stops the project's own entry point
from deleting the project's headline result.

## Status

| # | Plan | Category | Impact | Effort | Risk | Depends on | Status |
|---|---|---|---|---|---|---|---|
| [001](001-make-results-destroys-results-md.md) | `make_results.py` silently deletes hand-written `RESULTS.md` sections | correctness / data loss | HIGH | S | LOW | — | **DONE — awaiting merge** |
| [002](002-lora-seeding-and-run-provenance.md) | Tier-3 LoRA result is not reproducible: adapter init is unseeded | reproducibility | HIGH | S | LOW | 001 ✅ | **DONE — awaiting merge + GPU re-run** |
| [003](003-characterization-tests-for-the-metric-core.md) | No tests for the numeric core every published claim rests on | test coverage | HIGH | M | LOW | — | **DONE — awaiting merge** (47 tests green) |
| [004](004-consolidate-duplicate-topk-metrics.md) | Three copies of the top-k metric; Tier-1 deltas compare across two | tech debt | MED | S | MED | 003 ✅ | **DONE — awaiting merge** (48 tests; published numbers unmoved) |

### 001 — executed, reviewed, APPROVED (not merged)

Branch `worktree-agent-a75bd84018f62da00` at
`/home/user01/TITAN/.claude/worktrees/agent-a75bd84018f62da00`, two commits on top of `9d56a69`:

- `959b305` — `make_results.py`: merge sections into `RESULTS.md` instead of overwriting
- `da88ab8` — `merge()`: always separate sections with exactly one blank line

Reviewer-verified (re-run independently, not taken from the executor's report): Tier-3 section
preserved with its 52 content lines byte-identical; **no section lost**; BRACS baseline section
added (7 total); regeneration is a true no-op against the committed state (`git status` clean
after re-running); 4/4 tests pass. Scope clean — only `make_results.py` and the new
`validation/tests/test_make_results.py` were edited; `RESULTS.md` changed only as the *output* of
running the fixed script. No experiment script touched, no number changed.

**Merging is the user's call.** `git merge worktree-agent-a75bd84018f62da00` from `main`.

Two follow-ons this surfaced, both already covered elsewhere:
- `plans/` is untracked, so a worktree executor cannot read its own plan — plans must be inlined
  into the dispatch prompt (or `plans/` committed).
- Plan 002 edits `make_results.py` too (the `_provenance` skip in the subtasks loop) and must be
  rebased onto this branch, not onto `9d56a69`.

### 002 — executed, reviewed, APPROVED (not merged) — ⚠️ needs a GPU re-run to finish

Branch `worktree-agent-a530acf40f76edb2e` at
`/home/user01/TITAN/.claude/worktrees/agent-a530acf40f76edb2e`, one commit (`1fe057d`) on `da88ab8`.

Reviewer-verified independently. The executor's own "proof" that the bug existed was weak (its
pre-fix test failed with `AttributeError: no attribute 'reset_lora'` — which proves only that the
function didn't exist yet, not that the seeding was broken), so the reviewer demonstrated it
directly instead. Running the **old** init path (`get_peft_model`, unseeded) in three separate
processes yields three different adapters — `27c05720` / `506a75dd` / `bde6be50`. Running the
**new** path (`torch.manual_seed(k)` → `reset_lora`) yields `38767c71` in all three, and seed 1's
adapter is now identical whether or not seed 0 ran first. Both defects — the ambient-RNG init and
the entangled seed chain — are confirmed real and confirmed cured.

Also verified: seed loop order is `manual_seed(seed)` → `reset_lora` → `train_lora`; `train_lora`
no longer seeds; exactly two real `torch.manual_seed(` invocations remain (`train_linear_map:271`
untouched, `mode_lora:532`); cuDNN determinism on; 8/8 tests pass (002's 4 + 001's 4, no
regression); the `_provenance` guard genuinely protects `make_results.py` (injected the key — it
survives; without the guard it `KeyError`s) and does not leak into `RESULTS.md`.

**⚠️ The Tier-3 numbers in `RESULTS.md` are still the old, unreproducible ones.** The code is
fixed; the *record* is not. Finishing this finding requires the author's GPU re-run (plan 002 §9):
re-run `finetune_bracs_lora.py --mode lora`, update the Tier-3 table, and consider `--seeds 5..10`
while re-running, since Acc@1's seed std (0.015) is half its effect size (+0.032). If the re-run
does not reproduce LoRA > baseline on Acc@3, that is a real scientific result — report it, do not
tune seeds to recover the old numbers.

**Merging is the user's call.** `git merge worktree-agent-a530acf40f76edb2e` from `main`.

### 003 — executed; the suite immediately earned its keep

Branch `worktree-agent-a617d1c2de2c54c1c`. 46 tests written; **45 passed, 1 failed on the first
run** — and the failure was a genuine finding, correctly escalated by the executor rather than
smoothed away. Adjudicated by the reviewer as follows.

**The finding.** `jaccard_distance` can return values slightly **below 0** (measured worst case
−2.4e-4), violating its own `[0,1]` contract. Root cause: `build_V` stores the pooled N×N
neighbour-weight matrix as **float16** (`retrieval_tcga_ot_kreciprocal.py:88`, recast at `:114`).
Rows are normalized to sum to 1, but fp16 quantization plus `smooth_V`'s averaging pushes some row
sums fractionally *above* 1; since jaccard's self-similarity term **is** that row sum,
`1 − t/(2−t)` dips below zero. Confirmed reproducible, and the negatives reach the **ranked gallery
columns**, not just the diagonal.

**Adjudication: real, but benign — and the test's tolerance was the mistake, not the code.**
1. The fp16 storage is **by design**, mirroring Zhong et al.'s reference implementation, where it
   is a memory optimization (the real pooled matrix is 9,574² — ~733 MB fp32 vs ~183 MB fp16).
2. It **cannot reach any published number**. The sweep selects **λ=1.0**, and
   `final = jac*(1−λ) + original_dist*λ` multiplies jaccard by **exactly zero** there. That is also
   why the repo's λ=1.0 sanity check passes exactly (`lambda1_sanity_pass: true`).
3. It **cannot have flipped the selection** either: λ=1.0 beat the best non-trivial config
   (`k1=30 k2=6 λ=0.7`) by 0.0043 Acc@3 (~6 queries of 1348), and a 2.4e-4 *distance* perturbation
   would have to make re-ranking *better* — whereas every λ<1 config scored *worse*.

So `RESULTS.md`'s Tier-1 k-reciprocal conclusion stands untouched. The plan's `1e-6` bound was
wrong for a knowingly-fp16 intermediate; corrected to fp16 epsilon (1e-3), with the reasoning
written into the test docstring, plus a new test pinning that λ=1.0 is *identically* jaccard-free
(feed the term garbage — the ranking must not move).

Worth stating plainly: this is the first time anything in this repo has been able to tell its
author a true thing about the code that the author did not already believe. That is the whole
argument for plan 003.

### 004 — first attempt STOPPED; it exposed a defect in plan 003 that is now merged on `main`

The executor refused to run plan 004 and was right twice over.

**Defect 1 — plan 004 contradicted itself.** Step 2 said delete `topk_metrics`; §6 said
`validation/tests/**` was out of scope. But `test_metric_implementations_agree.py` imports
`topk_metrics` at module scope, so both could not hold. Root cause: an *"these three
implementations agree"* test is **self-destroying** — after consolidation there is no second
implementation left to compare against. The guarantee must be carried across the refactor by
pinning **outputs**, not implementations. Fixed: that test file is now in scope, and a `GOLDEN`
table (captured by the reviewer from the pre-refactor code) pins the survivor to exactly what the
three agreed on.

**Defect 2 — ⚠️ plan 003's test fixture is VACUOUS, and it is on `main` right now.** Found while
fixing Defect 1: capturing golden values from the three implementations returned **`1.0` for every
metric on all 20 seeds**. `_synthetic` places class centres at `rng.normal(...) * 3` against unit
noise, which separates the classes perfectly. So `test_eval_ranking_matches_topk_metrics` and
`test_agreement_holds_across_many_random_problems` have been asserting `1.0 == 1.0` — they would
pass against a badly broken implementation. The fixture's own comment ("metrics are non-trivial
(not all 0, not all 1)") is **false**. The reviewer wrote it.

This is the sharpest lesson of the whole exercise, and it is the one plan 003 §8 warned about in
the abstract while committing it in practice: *a test that passes proves nothing until you know
what it would take to make it fail.* A green suite is not evidence; a green suite on a fixture that
can distinguish right from wrong is.

Fix (in plan 004, Step 0.5): centres `* 3` → `* 0.45`. At that setting acc@1≈0.56 / acc@3≈0.84 /
mvacc@3≈0.58, **no seed saturates**, `mvacc@3 ≠ acc@3`, and the majority-vote **tie-break branch
fires on 18% of queries (90/500)** rather than never. All three implementations still agree there —
so the agreement now means something. A `test_the_fixture_is_actually_discriminative` tripwire is
added so it cannot silently regress.

**Note for the maintainer:** until plan 004 lands, `main` carries tests that assert nothing.
Nothing published is wrong because of it — but the suite is weaker than its green tick implies.

### 004 — second attempt: APPROVED (not merged)

Branch `worktree-agent-aaeb31b756da2414b`, commits `40f24c2` + `b1de561` on `b803ba5`.

The metric is now implemented **once** (`retrieval_common.eval_ranking`); `topk_metrics` and the
inline `subtasks` loop are gone; `BASELINE_REF`'s provenance is documented; and the k-reciprocal
λ=1.0 check is now a hard `sys.exit` *before* `save_results`, so a run that fails its own identity
test can no longer write a result file.

**Every published number is unchanged** — `ALL PUBLISHED RETRIEVAL NUMBERS UNCHANGED`
(TCGA-OT 0.8717 / 0.7812 and all five sub-task rows, re-derived by actually re-running the scripts
offline). 48 tests green.

**A second vacuous fixture was found and fixed.** `_data` in `test_retrieval_postproc.py` also
saturated (5/5 seeds at 1.0), which had gutted **`test_lambda_one_reproduces_raw_cosine_exactly`** —
the crown jewel, since the λ=1.0 identity is the only correctness check this repo ever had, and
Step 5 had just promoted it to a hard abort. Centres `* 2` → `* 0.5`. It now asserts between
non-trivial values (acc@1 0.5, acc@3 0.8) and **still passes**, so k-reciprocal genuinely does
reduce to plain cosine — it simply had no evidence behind it before.

**Reviewer mutation tests — the tests now bite:**

| sabotage | old (saturated) fixture | fixed fixture |
|---|---|---|
| `eval_ranking` tie-break `cands[0]` → `cands[-1]` | caught on **0/20** seeds | caught on **15/20** |
| λ nudged 1.0 → 0.99 | **passed — vacuous** | **fails — caught** |
| full-rank PCA truncated by one component | **passed — vacuous** | **fails — caught** |

**Saturation sweep of the remaining suite (reviewer, closing the executor's open question):**
only `test_metric_implementations_agree.py` and `test_retrieval_postproc.py` ever used a
class-structured random fixture, and both are now fixed with `test_the_fixture_is_actually_
discriminative` tripwires. The other five files assert on hand-computed literals
(`test_ranking_metrics`), deterministic tensors (`test_bracs_patch_geometry`, `test_lora_seeding`),
structural properties (`test_proxy_anchor`), or strings (`test_make_results`) — **immune by
construction.** The audit is closed.

**Merging is the user's call.** `git merge worktree-agent-aaeb31b756da2414b` from `main`.

## 1. Findings that became plans

**001 — `make_results.py` destroys `RESULTS.md`.** `validation/run_all.sh` (the documented entry
point) ends by running `make_results.py`, which rebuilds `RESULTS.md` from scratch and has no
handler for `finetune_bracs_lora_step3.json`. Verified by regenerating into a scratch dir: the
output is **58 lines** where the committed file is **111**, and the entire `## Retrieval Tier-3`
section — the repo's only *positive* result, ~50 lines of hand-written analysis that no JSON can
reproduce — is gone. `retrieval_bracs.json` and `retrieval_camelyon16.json` are also orphaned:
result files on disk that no section ever reads.

**002 — the Tier-3 seeds are not seeds.** `get_peft_model()`
(`finetune_bracs_lora.py:498`) kaiming-inits every `lora_A` from torch's global RNG, and the first
`torch.manual_seed` does not run until `train_lora` at line 425. Torch seeds its default generator
from OS entropy — confirmed: `torch.initial_seed()` returns a different value on every process.
Seeds 1 and 2 are worse: the re-init sits at the *bottom* of the loop (lines 528-532), so each
seed's adapter is a function of the *previous* seed's training. The reported `0.5368 ±0.0149` and
the "seed 1's CI excludes zero" claim describe one unrepeatable chain. The finding itself (LoRA >
baseline on Acc@3, tight spread) will very likely survive a re-run — but the published `±` cannot
be regenerated by anyone, including its author. Note the contrast that makes this a bug and not a
choice: the Step-0 *control* path (`train_linear_map:271`) seeds correctly.

**003 — zero tests.** 4,649 lines, no `tests/`, no CI, `pytest` not even installed. The numeric
core is all hand-rolled: `eval_ranking`, k-reciprocal's `build_V`/`jaccard_distance`, αQE/DBA's
`_expand`, `fit_pca`/`make_transform`, `ProxyAnchor`, and `_pad_grid_to_2x2` (which sits under
*every* BRACS embedding). The only correctness check anywhere is k-reciprocal's λ=1.0 identity —
and it is `print`ed, not enforced; on `FAIL` the script saves its results and reports a conclusion
anyway. All of it is pure numpy/torch: testable on CPU, offline, in seconds.

**004 — three copies of the metric.** `retrieval_common.eval_ranking` (BRACS + all Tier-1),
`retrieval_tcga_ot.topk_metrics` (the headline baseline), and an inline loop in
`subtasks_tcga_ot.retrieval` (the five sub-task rows). They agree today — copy 3 only by accident
of `Counter.most_common`'s stable sort, which is not part of its documented contract. Meanwhile
`retrieval_common.BASELINE_REF = {0.8717, 0.7812}` hardcodes numbers produced by *copy 2*, while
every Tier-1 delta is computed with *copy 1*. Nothing would catch a divergence.

## 2. Direction — where to take the science next

Options for the maintainer, not defects. Ordered by information-per-GPU-hour.

**a) Test the ceiling directly: LoRA the patch encoder, not just the aggregator.**
`RESULTS.md` already localizes the failure: ADH/UDH/DCIS retrieve each other 56–80% of the time,
LoRA moves DCIS (+0.106) and N (+0.074) but leaves ADH/UDH flat, and the write-up concludes "that
inseparability is **CONCH-level** … which an aggregator LoRA cannot fix." That is a falsifiable
claim about where the bottleneck lives, and nothing in the repo tests it. A LoRA on CONCHv1.5's
last blocks — same Proxy-Anchor loss, same patient-disjoint BRACS protocol, same val-selection —
would settle it. Cost: the patch cache no longer suffices (CONCH must run in the training loop),
so this is materially more expensive than Step 3. Trade-off: it is the one experiment whose
*negative* result would be as informative as its positive one.

**b) Run Tier-3 at WSI scale, on features that already exist.**
The repo's own hedge is that BRACS ROIs are a weak stress test — 33% are ≤2 patches, so naive
mean-pooling *ties* TITAN's aggregator (0.525 vs 0.505 Acc@1) and "the ceiling is low by
construction." The stated expectation is that the payoff is larger on gigapixel WSIs where the
aggregator does real work. There is already a local cohort with exactly the right inputs:
CAMELYON16's `h5_files/conch_v1_5/` holds CONCHv1.5 **patch features + coords** for 100 slides
(72 labeled) — precisely what `encode_slide_from_patch_features` consumes, at up to ~23k patches
per slide. Caveats to design around: only 72 labeled slides and 2 classes (thin for a retrieval
DB), and the features are 512px@40x where CONCH expects 20x — the magnification caveat already
flagged in `RESULTS.md`, and the exact error `extract_bracs_features.py` was written to avoid.
Worth a spike to decide if 72 slides can support the claim at all before committing to it.

**c) More seeds, once 002 lands.**
Acc@1's seed std (0.015) is nearly half its effect size (+0.032), which is why the paired
bootstrap only clears zero for one of three seeds. Acc@3 (+0.042, std 0.002) is solid; Acc@1 is
"positive-but-marginal per-seed" by the repo's own admission. With reproducible seeding, going to
5–10 seeds is the cheapest honest way to firm that up — or to retract it. Do this *with* the 002
re-run, not as a separate GPU booking.

**d) Not recommended: optimizing the LoRA training loop.**
`train_lora` re-embeds the whole 3,657-ROI database every epoch, one ROI per forward pass in a
Python loop, to compute the val-retrieval selection signal — this dominates the ~40 min runtime.
It is tempting. But batching would change the numerics of the very selection step that picks the
reported checkpoint, for a saving of GPU-minutes on a run that happens a handful of times. The
juice is not worth the squeeze; leave it.

## 3. Considered and rejected

Recorded so they are not re-audited next run.

- **`subtasks_tcga_ot.retrieval` tie-break differs from `eval_ranking`.** Looked like a real bug.
  It is not: `Counter.most_common` sorts by count with a stable sort, so ties preserve insertion
  order, and `retr` is already in descending-similarity order — the same answer `cands[0]` gives.
  Downgraded from "bug" to "fragile", and folded into plan 004 as a reason to consolidate rather
  than as a defect in its own right.
- **Path traversal in `download_bracs.py`.** `download_file` builds
  `dest / r.official_split / r.label / Path(r.remote).name` from server-controlled directory names.
  Technically a traversal surface. Rejected: `Path(...).name` strips directories from the one
  attacker-influenced component that matters, the host is an academic FTP server the user names
  explicitly in `.env`, and this is a research download script. Not worth a plan.
- **`.env` in the repo root.** Contains a live `HF_TOKEN`. Correctly listed in `.gitignore` and
  confirmed untracked (`git ls-files --error-unmatch .env` → not found). No secret is committed.
  Not a finding.
- **k-reciprocal / whitening / αQE / DBA math.** Read line-by-line against Zhong et al. (2017) and
  the standard αQE/DBA formulations. `precompute`'s column-max normalization, `build_V`'s 2/3
  overlap rule, `smooth_V`, and the `1 - temp_min/(2 - temp_min)` Jaccard all match the reference
  implementations. `_expand`'s self-exclusion and clamping are correct. The negative Tier-1 result
  is not an artifact of a bug. (Plan 003 pins all of this with tests anyway — reading is not the
  same as testing.)
- **`ProxyAnchor.forward`.** Matches Kim et al. (CVPR 2020): positive term divided by |P⁺|
  (classes present in the batch), negative term by |P| (all classes). Correct.
- **`_pad_grid_to_2x2`.** Traced the single-tile, horizontal-strip, vertical-strip and already-2×2
  cases; all produce a valid ≥2×2 grid, and the "the only zero rows are injected pads" invariant is
  asserted at cache time (`cache_bracs_patch_features.py:72-75`). Correct. Tested in plan 003
  because a regression here would silently corrupt every BRACS number.
- **Per-ROI `torch.cuda.empty_cache()`** (in `common.py:154`, `extract_bracs_features.py:131`,
  `cache_bracs_patch_features.py:120`, `finetune_bracs_lora.py:89`). A device sync + allocator
  flush on every ROI, in code that runs over 4,539 ROIs. Real, but it is on the one-time extraction
  paths, not the training loop, and the comment in `common.py:110-114` shows the OOM behaviour was
  reasoned about deliberately. Wall-clock only, no effect on any number. Not worth the risk.

## 4. What was NOT audited

- The `titan/` package itself (`utils.py`, `finetune.py`, `eval_linear_probe.py`) — upstream
  MahmoodLab code, not this project's to change. Read only far enough to confirm the validation
  scripts call it correctly (`get_eval_metrics`, `bootstrap`, `seed_torch`, the C-grid).
- `notebooks/` — upstream demos, unmodified.
- The correctness of the **published TITAN paper numbers** themselves (0.704 / 0.880 / 0.807). The
  repo reproduces them to within a plausible margin; that is the strongest available evidence the
  pipeline is right, and it is a meaningful check that the audit did not need to redo.
- Anything requiring the 51.8 GB BRACS download, a GPU run, or an HF token. Every finding above was
  established by reading source, inspecting the committed result JSONs, and running read-only
  checks on CPU.

## 5. Minor items — fix inline, not worth a plan

- **`.gitignore` misses `*.log`.** `validation/results/step3_run.log` is untracked and unignored;
  it will be swept into the next `git add -A`. (Folded into plan 002, step 7.)
- **`finetune_bracs_lora.py`'s module docstring is stale.** It says "Currently implements: `--mode
  baseline` … (bf16, LoRA disabled)". The file now implements four modes, the default `--dtype` is
  `fp16`, and training is fp32 (per `RESULTS.md`: "the TITAN RTX has no bf16"). `bf16` is still in
  `DTYPES` and will silently fail or fall back on this Turing card. Rewrite the docstring to
  describe all four modes; consider dropping `bf16` from the `--dtype` choices.
- **`.env.example` documents `BRACS_WSI_DIR`**, but `download_bracs.py:422` reads `BRACS_DIR`.
  One-word fix in `.env.example`.
