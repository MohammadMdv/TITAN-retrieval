# Plan 002 — The Tier-3 LoRA result is not reproducible: the adapter init is unseeded

**Written against commit:** `9d56a69` · **re-stamped and drift-checked at `da88ab8`** (post-001 merge)
**Category:** correctness / scientific reproducibility
**Impact:** HIGH · **Effort:** S · **Risk of fix:** LOW (but it *will* change the reported numbers — see §9)
**Depends on:** plan 001 — **DONE and merged** (commits `959b305`, `da88ab8`).

> **Drift check at `da88ab8`:** `finetune_bracs_lora.py`, `common.py` and `.gitignore` are
> **untouched** by plan 001, so every excerpt below is still accurate. The only file 001 changed
> that this plan also edits is `make_results.py` — and the `for name, r in sub.items():` loop that
> Step 6 targets survived 001 unchanged; it now sits at **line 180** inside the
> `generated.append(("Sub-tasks — …", lines))` block. Step 6 applies verbatim.

---

## 1. Why this matters

`RESULTS.md` reports the repo's headline positive result as:

> **Block-LoRA + Proxy-Anchor** — Acc@1 **0.5368 ±0.015**, Acc@3 **0.7573 ±0.002** (3 seeds)
> … the **val-selected best seed is the significant one** (Δ+0.053, CI [+0.014, +0.089] excludes 0)

Those `±` figures and that significance claim are seed statistics. **They cannot be reproduced,
because the LoRA adapter's initialization is never seeded.**

Three separate defects in `validation/finetune_bracs_lora.py`, in order of severity:

**(a) Seed 0's adapter is initialized from the process's ambient RNG.**
`mode_lora()` builds the adapter at line 498:

```python
# validation/finetune_bracs_lora.py:496-498
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=targets, bias="none")
    peft_model = get_peft_model(model, cfg)      # <-- PEFT kaiming-inits every lora_A HERE
```

`get_peft_model` draws `lora_A` from `torch.nn.init.kaiming_uniform_`, i.e. from torch's global
RNG. The first `torch.manual_seed` in the whole program does not happen until line 425 — inside
`train_lora`, which is not called until line 516. So by the time anything is seeded, `lora_A` is
already set from an **unseeded** generator. PyTorch seeds its default generator from OS entropy,
which is verified here:

```
$ .venv/bin/python -c "import torch; print(torch.initial_seed())"   # run 3x
16398001705143910732
9466708432963321776
1588140960277928583
```

Different every process. So every invocation of `--mode lora` starts from a different adapter.

**(b) Seeds 1 and 2 inherit seed 0's RNG state.**
The re-init between seeds happens at the *end* of each loop iteration:

```python
# validation/finetune_bracs_lora.py:527-532
        seeds_out.append({...})
        # reset adapter to zero for the next seed (fresh init)
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                nn.init.zeros_(p)
            elif "lora_A" in n:
                nn.init.kaiming_uniform_(p, a=np.sqrt(5))
```

This runs *after* seed *k*'s training, consuming whatever RNG state that training left behind,
and *before* seed *k+1*'s `torch.manual_seed(k+1)` (line 425). So seed 1's adapter is a function
of seed 0's entire training run, not of the integer `1`. The seeds are not independent draws;
they are one entangled chain.

**(c) There is no determinism control at all** — no `cudnn.deterministic`, no
`cudnn.benchmark = False` (grep across `validation/` finds neither).

Note the contrast that makes this a clear bug rather than a design choice: the Step-0 **control**
path is seeded correctly (`train_linear_map`, line 271, seeds *before* it builds anything), and
its `nn.Linear` is identity-initialized anyway (lines 278-279), so the controls *are*
reproducible. Only the headline LoRA path is not.

### What is and is not invalidated

Be precise about this — it matters for how the result gets written up:

- **The finding itself almost certainly survives.** Three independent (if uncontrolled) inits all
  beat the baseline on Acc@3 with a tight spread (0.754 / 0.760 / 0.7596), and the LoRA-off
  baseline it is measured against is deterministic. Randomly-initialized adapters are exactly
  what "3 seeds" is meant to sample.
- **The specific numbers are not re-derivable.** `0.5368 ±0.0149`, `0.7573 ±0.0022`, and
  "seed 1 has CI [+0.014, +0.089]" describe *one* unrepeatable chain of three inits. Re-run the
  script today and you get three different adapters, hence different per-seed rows, a different
  std, and possibly a different "best" seed.

That is the defect: a published `±` that nobody — including the author, tomorrow — can regenerate.

## 2. The fix, in one sentence

Seed exactly once per seed index, *before* the adapter for that seed exists, so that seed *k*'s
run is a pure function of *k*; then stamp every result JSON with enough provenance (git SHA,
torch/GPU, argv) to tie a number back to the code that produced it.

## 3. Current state — the three code sites

### Site 1 — `mode_lora()`, adapter construction (lines 491–503)

```python
    device = get_device()
    model, device = load_titan(device)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = block_targets(model)
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=targets, bias="none")
    peft_model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] LoRA on {len(targets)} Linears; trainable params={n_train:,}")

    print("[lora] preloading patch cache to GPU ...")
    cache = preload_cache(tr_ids + va_ids + te_ids, device)
```

### Site 2 — `mode_lora()`, the seed loop (lines 513–532)

```python
    seeds_out = []
    for seed in range(args.seeds):
        print(f"\n[lora] --- seed {seed} ---")
        best_sig = train_lora(model, peft_model, cache, tr_ids, ytr, ctr, va_ids, yva, cva,
                              classes, tile, device, seed, args)
        Zdb = embed_ids(model, cache, tr_ids, tile)
        Zte = embed_ids(model, cache, te_ids, tile)
        m, pc = retrieval_metrics(Zdb, ytr, ctr, Zte, yte, cte, f"lora-s{seed}")
        lora_ok = top1_correct(Zdb, ytr, Zte, yte)
        paired = paired_bootstrap(base_ok, lora_ok)
        print(f"[lora] seed {seed} test: " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
              + f"  | dAcc@1={paired['delta_acc@1']:+.4f} CI{paired['ci95']}")
        seeds_out.append({"seed": seed, "val_sig": best_sig, "metrics": m,
                          "per_class_acc@1": pc, "paired_vs_loraoff": paired})
        # reset adapter to zero for the next seed (fresh init)
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                nn.init.zeros_(p)
            elif "lora_A" in n:
                nn.init.kaiming_uniform_(p, a=np.sqrt(5))
```

### Site 3 — `train_lora()`, its own seeding (lines 421–431)

```python
def train_lora(model, peft_model, cache, tr_ids, ytr, ctr, va_ids, yva, cva, classes, tile,
               device, seed, args):
    """Train Block-LoRA + Proxy-Anchor, selecting on val retrieval; keep best-epoch adapter."""
    from copy import deepcopy
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    C = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    ytr_idx = np.array([cls_idx[c] for c in ytr])
    tr_ids = np.asarray(tr_ids)
```

## 4. Steps

### Step 1 — add a `reset_lora` helper

In `validation/finetune_bracs_lora.py`, add this immediately **after** `block_targets()`
(around line 373):

```python
def reset_lora(peft_model):
    """Re-init the adapter to PEFT's default state: lora_A ~ kaiming_uniform, lora_B = 0.

    Called at the START of every seed, right after that seed's torch.manual_seed, so seed k's
    adapter is a pure function of k -- independent of the ambient process RNG and of whatever
    earlier seeds did. (PEFT does this init inside get_peft_model, which runs before any seeding,
    so relying on it makes the run unreproducible.)
    """
    for n, p in peft_model.named_parameters():
        if "lora_B" in n:
            nn.init.zeros_(p)
        elif "lora_A" in n:
            nn.init.kaiming_uniform_(p, a=np.sqrt(5))
```

### Step 2 — seed once per seed, before the adapter is reset

In `mode_lora()`, change the head of the seed loop (site 2) to:

```python
    seeds_out = []
    for seed in range(args.seeds):
        print(f"\n[lora] --- seed {seed} ---")
        torch.manual_seed(seed)          # the ONE seeding point for this seed: adapter init,
        np.random.seed(seed)             # proxy init, LoRA dropout and PK sampling all follow
        reset_lora(peft_model)           # fresh adapter, drawn from the just-seeded generator
        best_sig = train_lora(model, peft_model, cache, tr_ids, ytr, ctr, va_ids, yva, cva,
                              classes, tile, device, seed, args)
        ...
```

and **delete** the trailing 5-line reset block (lines 527–532, the
`# reset adapter to zero for the next seed` loop). Keep the `seeds_out.append(...)` above it.

### Step 3 — remove the now-duplicate seeding inside `train_lora`

In `train_lora` (site 3), delete this line:

```python
    torch.manual_seed(seed); np.random.seed(seed)
```

and keep `rng = np.random.default_rng(seed)` (that one is already correct and independent).

Rationale: seeding is now the caller's job and happens once, before the adapter exists. Leaving
the call in would *rewind* the generator after `reset_lora` had already drawn from it, which
still works but makes the adapter and the Proxy-Anchor proxies share an RNG prefix — pointless
and confusing. One seeding point, at the top of the loop.

> `train_lora` keeps its `seed` parameter — it is still used for `np.random.default_rng(seed)`.

### Step 4 — turn on cuDNN determinism

At the top of `mode_lora()`, right after `device = get_device()`:

```python
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Do **not** add `torch.use_deterministic_algorithms(True)`. It raises `RuntimeError` on several
ops used inside TITAN's attention path and would break the run outright; the cuDNN flags are the
safe subset. Residual GPU float non-associativity means two runs may still differ in the last
few decimals — that is expected and is why step 5 exists.

### Step 5 — stamp provenance on every result JSON

The result JSONs currently record metrics but nothing about *what produced them* — no commit, no
torch version, no GPU, no command line. Fix this once, centrally, in `validation/common.py`.

Add above `save_results` (which is at line 102):

```python
def _provenance():
    """What produced this JSON: enough to tie a number back to the code and box that made it."""
    import platform
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=TITAN_ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=TITAN_ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        sha, dirty = None, None
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "argv": sys.argv,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }
```

and change `save_results` to stamp it:

```python
def save_results(name, obj):
    path = RESULTS_DIR / name
    if isinstance(obj, dict):
        obj = {**obj, "_provenance": _provenance()}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {path}")
    return path
```

### Step 6 — make `make_results.py` ignore the `_provenance` key

**This step is mandatory and is why plan 001 must land first.** `make_results.py` iterates the
top level of `subtasks_tcga_ot.json` and assumes every key is a task:

```python
# validation/make_results.py, subtasks block
        for name, r in sub.items():
            lines.append(f"| {name} | {len(r['classes'])} | {r['n']['test']} | ...")
```

After step 5 that dict gains a `_provenance` key, and `r['classes']` will raise `KeyError`.
Change the loop to skip private keys:

```python
        for name, r in sub.items():
            if name.startswith("_"):        # _provenance stamp, not a task
                continue
            lines.append(f"| {name} | {len(r['classes'])} | {r['n']['test']} | ...")
```

Audit the other `load(...)` consumers in `make_results.py` while you are there: all of them index
known keys (`ot["linear_probe"]`, `ret['acc@3']`, `w["selected"]`, …) rather than iterating, so
the subtasks loop is the only site that breaks. Confirm this by reading each block; do not assume.

### Step 7 — gitignore run logs

`validation/results/step3_run.log` is currently untracked and **not** ignored (`.gitignore`
covers `*.out` but not `*.log`), so it will be swept into the next `git add -A`. Add to
`.gitignore`, under the existing `# Log captures` heading:

```
*.out
*.log
```

## 5. Verification

### 5a — unit tests (no GPU, no network, no BRACS data)

Create `validation/tests/test_lora_seeding.py`. It builds a tiny PEFT model on a toy `nn.Module`
so it runs on CPU in under a second — it does **not** load TITAN.

```python
"""Seed k's LoRA adapter must be a pure function of k."""
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import finetune_bracs_lora as fl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _peft():
    return get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=["fc1", "fc2"],
                                             bias="none"))


def _adapter_state(m):
    return {n: p.detach().clone() for n, p in m.named_parameters() if "lora_" in n}


def test_reset_is_deterministic_given_the_seed():
    """Same seed -> same adapter, on two independently constructed models."""
    a, b = _peft(), _peft()
    torch.manual_seed(7); fl.reset_lora(a)
    torch.manual_seed(7); fl.reset_lora(b)
    sa, sb = _adapter_state(a), _adapter_state(b)
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_different_seeds_give_different_adapters():
    a, b = _peft(), _peft()
    torch.manual_seed(0); fl.reset_lora(a)
    torch.manual_seed(1); fl.reset_lora(b)
    diff = [k for k in _adapter_state(a)
            if "lora_A" in k and not torch.equal(_adapter_state(a)[k], _adapter_state(b)[k])]
    assert diff, "seeds 0 and 1 produced identical lora_A -- reset is not drawing from the RNG"


def test_seed_k_does_not_depend_on_earlier_seeds():
    """The real bug: seed 1's adapter must be the same whether or not seed 0 ran first."""
    m = _peft()
    torch.manual_seed(1); fl.reset_lora(m)
    alone = _adapter_state(m)

    m2 = _peft()
    torch.manual_seed(0); fl.reset_lora(m2)
    _ = torch.randn(1000)                       # stand in for seed 0's training consuming RNG
    torch.manual_seed(1); fl.reset_lora(m2)     # seed 1 re-seeds first -> must be unaffected
    after = _adapter_state(m2)

    for k in alone:
        assert torch.equal(alone[k], after[k]), k


def test_lora_B_starts_at_zero():
    """B=0 means the adapter is a no-op at init: an untrained LoRA == the frozen encoder."""
    m = _peft()
    torch.manual_seed(3); fl.reset_lora(m)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            assert torch.count_nonzero(p) == 0, n
```

Run: `.venv/bin/python -m pytest validation/tests/test_lora_seeding.py -q` → **4 passed**.

`test_seed_k_does_not_depend_on_earlier_seeds` is the one that fails against the current code and
passes after the fix. Confirm that: `git stash` the change, watch it fail, unstash.

### 5b — provenance smoke test (no GPU)

```bash
cd /home/user01/TITAN/validation && ../.venv/bin/python -c "
from common import save_results
import json, tempfile, pathlib
p = save_results('_provenance_smoke.json', {'x': 1})
d = json.loads(pathlib.Path(p).read_text())
assert d['x'] == 1
pr = d['_provenance']
assert pr['git_sha'] and pr['torch'] and pr['python'], pr
print('provenance OK:', pr)
pathlib.Path(p).unlink()      # clean up; do not leave a stray file in results/
"
```

### 5c — the real run (GPU, ~40 min; the author should do this, not the executor)

```bash
cd /home/user01/TITAN/validation
../.venv/bin/python finetune_bracs_lora.py --mode lora --seeds 3 2>&1 | tee results/step3_rerun.log
```

Then run it **a second time** and diff the per-seed test metrics. With the fix, the two runs
should agree to ~3 decimal places (residual GPU float non-associativity), where today they would
differ substantially. Compare with:

```bash
../.venv/bin/python -c "
import json; d=json.load(open('results/finetune_bracs_lora_step3.json'))
for s in d['per_seed']: print(s['seed'], {k: round(v,4) for k,v in s['metrics'].items()})
print('mean', {k: round(v,4) for k,v in d['test_mean'].items()})
print('std ', {k: round(v,4) for k,v in d['test_std'].items()})
"
```

## 6. Done criteria (machine-checkable)

- [ ] `.venv/bin/python -m pytest validation/tests/test_lora_seeding.py -q` → 4 passed.
- [ ] `grep -n 'torch.manual_seed' validation/finetune_bracs_lora.py` shows exactly **two**
      call sites: one in `train_linear_map` (unchanged, line ~271) and one at the top of
      `mode_lora`'s seed loop. **None** inside `train_lora`.
- [ ] `grep -n 'reset adapter to zero for the next seed' validation/finetune_bracs_lora.py`
      returns nothing (the trailing reset block is gone).
- [ ] `grep -n 'cudnn.deterministic' validation/finetune_bracs_lora.py` returns 1 hit.
- [ ] The provenance smoke test (5b) prints a non-null `git_sha`.
- [ ] `.venv/bin/python -m pytest validation/tests -q` → all green (nothing else regressed).
- [ ] `grep -q '^\*\.log$' .gitignore` exits 0.

## 7. Scope

**In scope:** `validation/finetune_bracs_lora.py`, `validation/common.py` (`save_results` +
`_provenance` only), `validation/make_results.py` (the one `_`-key skip in the subtasks loop),
`.gitignore`, `validation/tests/test_lora_seeding.py` (new).

**Out of scope — do not touch:**
- `train_linear_map` (line 267) — its seeding is already correct; leave it exactly as it is. It
  is the working reference for what "correct" looks like here.
- The `ProxyAnchor` class, `pk_batches`, `cosine_lr`, the training loop body, and every
  hyperparameter (`r=8, alpha=16, lr=2e-4, K=8, epochs=20, patience=5`). This plan changes
  **which random numbers are drawn**, not the method. If you find yourself editing the loss or
  the schedule, stop.
- `validation/results/finetune_bracs_lora_step3.json` — the old, unreproducible record. Leave it
  on disk; §9 says what to do about it.
- `validation/RESULTS.md` — do not edit the Tier-3 numbers by hand. See §9.

## 8. Maintenance note

The rule this establishes: **seed immediately before the thing whose randomness you want to
control, never after.** The bug here was structural — the re-init sat at the *bottom* of the loop
body, which reads naturally ("clean up for the next iteration") but means the RNG state that
feeds it belongs to the previous iteration. Any future multi-seed loop in this repo should seed
at the top and construct at the top.

Watch for this in review whenever `get_peft_model`, `nn.init.*`, or a fresh `nn.Module` appears
outside a seeded region.

## 9. ⚠️ What to do about the already-published Tier-3 numbers

The fix changes which adapters get trained, so **the numbers in `RESULTS.md` §Tier-3 will move**
when the experiment is re-run. Do not paper over this.

1. Land the code fix.
2. Re-run 5c to produce a reproducible 3-seed record.
3. Update the Tier-3 table in `RESULTS.md` with the new per-seed numbers, and add a line noting
   the seeds are now reproducible (`git_sha` is in the JSON).
4. **Consider raising `--seeds` to 5 or 10 while you are re-running.** Acc@1's seed std (0.015)
   is half the effect size (+0.032), which is exactly why the paired test only clears zero for
   one of three seeds. With reproducible seeding, more seeds is now a cheap and honest way to
   firm up — or honestly retract — the Acc@1 claim. Acc@3 (+0.042, std 0.002) is unaffected by
   this concern and will hold.

If the re-run does **not** reproduce the qualitative finding (LoRA > baseline on Acc@3), that is
a genuine scientific result and must be reported as such — **do not** tune seeds or
hyperparameters to recover the old numbers. Stop and report back.

## 10. Escape hatches — STOP and report back if:

- `peft` names its adapter parameters something other than `lora_A` / `lora_B` in the installed
  version (this repo has `peft 0.13.2`, where the names hold). The `reset_lora` helper and the
  `best_state` filter in `train_lora` (line 463-464, `if "lora_" in k`) both depend on that
  string. If the names differ, every reset silently becomes a no-op — which would be *worse* than
  the current bug. `test_different_seeds_give_different_adapters` is the tripwire for this.
- `test_seed_k_does_not_depend_on_earlier_seeds` passes *before* you make any change. That would
  contradict this plan's premise; report it rather than proceeding.
- Adding the `_provenance` stamp breaks a `make_results.py` block other than the subtasks loop.
  Report which; do not start rewriting the reporting logic.
