# Plan 001 — `make_results.py` silently deletes hand-written sections of `RESULTS.md`

**Written against commit:** `9d56a69`
**Category:** correctness / data loss
**Impact:** HIGH · **Effort:** S · **Risk of fix:** LOW
**Depends on:** nothing. **Blocks:** plan 002 (which also edits `make_results.py`).

---

## 1. Why this matters

`validation/RESULTS.md` is this repository's primary scientific artifact — it is the file a
reader opens to see what was found. It is 111 lines and contains six `##` sections.

`validation/make_results.py` regenerates that file **from scratch** on every run:

```python
# validation/make_results.py:134
    OUT.write_text("\n".join(lines))
```

It only knows how to generate **five** of the six sections. It has no handler for
`finetune_bracs_lora_step3.json`, so the section titled
**`## Retrieval Tier-3 — LoRA fine-tuning of the TITAN slide encoder (BRACS ROIs)`**
(lines 60–111 of `RESULTS.md`, added by commit `9d56a69`, the repo's headline result — the
only experiment in the project that *positively* improved retrieval) is simply not in its
output. Neither are the BRACS baseline (`retrieval_bracs.json`) or the CAMELYON16 retrieval
comparison (`retrieval_camelyon16.json`), both of which have result JSONs on disk that no
section ever reads.

`validation/run_all.sh` — the documented entry point in `validation/README.md` — ends with:

```bash
# validation/run_all.sh:33
$PY make_results.py
```

**So running the project's own top-level script deletes the project's headline result from its
own results document.** This is verified, not theoretical: regenerating `RESULTS.md` with the
current code produces a **58-line** file where the committed one is **111 lines**, and the
`## Retrieval Tier-3` heading is gone. Today the only thing protecting that section is that it
happens to be committed to git.

Worse, the Tier-3 section is not merely a table — it contains ~50 lines of hand-written
analysis that **cannot be regenerated from any JSON**: the per-class Acc@1 breakdown narrative,
the "Where the gains come from" paragraph, and the "Context — why this ROI result matters
despite being modest" discussion. Regenerating it is not an option; it must be *preserved*.

## 2. The fix, in one sentence

Make `make_results.py` **update** `RESULTS.md` instead of **overwriting** it: sections it knows
how to generate get regenerated; every other section already in the file is carried through
verbatim. Then add a generated section for the BRACS baseline, which currently has no home.

## 3. Current state — read these before editing

### `validation/make_results.py` (140 lines, entire structure)

```python
"""Aggregate per-experiment JSON results into RESULTS.md (compared to paper reference numbers)."""
import json
from pathlib import Path
from common import RESULTS_DIR

OUT = Path(__file__).resolve().parent / "RESULTS.md"


def load(name):
    p = RESULTS_DIR / name
    return json.load(open(p)) if p.exists() else None


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def delta(x, base):
    return f"{x - base:+.4f}"


def main():
    lines = ["# TITAN Validation Results", "",
             "Reference numbers from the TITAN Nature Medicine paper / repo README.", ""]

    smoke = load("smoke_slide_encoding.json")
    if smoke:
        lines += ["## Smoke test — slide encoding", ...]

    cam = load("classification_camelyon16.json")
    if cam:
        lines += ["## Classification — CAMELYON16 binary (tumor vs normal)", ...]

    ot = load("classification_tcga_ot.json")
    if ot:
        lines += ["## Classification — TCGA-OT (46-class, full WSIs)", ...]

    ret = load("retrieval_tcga_ot.json")
    if ret:
        lines += ["## Retrieval — TCGA-OT slide retrieval (patient-disjoint)", ...]

    # ---- Tier-1: training-free retrieval post-processing on frozen embeddings ----
    tier1 = [...]
    loaded = {n: load(f) for _, f in tier1 for n in [f]}
    if any(loaded.values()):
        lines += ["## Retrieval Tier-1 — training-free post-processing (negative result)", ...]

    sub = load("subtasks_tcga_ot.json")
    if sub:
        lines += ["## Sub-tasks — harder patient-disjoint TCGA-OT sub-typing", "",
                  "| Task | #cls | test n | LP bacc | Ret Acc@1 | MVAcc@3 |", "|---|---|---|---|---|---|"]
        for name, r in sub.items():
            lines.append(f"| {name} | {len(r['classes'])} | {r['n']['test']} | "
                         f"{fmt(r['linear_probe'].get('bacc'))} | {fmt(r['retrieval']['acc@1'])} | "
                         f"{fmt(r['retrieval']['mvacc@3'])} |")
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"[make_results] wrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

The six section headings currently in `validation/RESULTS.md`, in order:

```
5:## Classification — CAMELYON16 binary (tumor vs normal)
14:## Classification — TCGA-OT (46-class, full WSIs)
22:## Retrieval — TCGA-OT slide retrieval (patient-disjoint)
30:## Retrieval Tier-1 — training-free post-processing (negative result)
50:## Sub-tasks — harder patient-disjoint TCGA-OT sub-typing
60:## Retrieval Tier-3 — LoRA fine-tuning of the TITAN slide encoder (BRACS ROIs)   <-- DESTROYED
```

### `validation/results/retrieval_bracs.json` (the orphaned result)

Produced by `validation/retrieval_bracs.py`. Shape (confirmed by reading `retrieval_bracs.py`
`main()` and `evaluate()`):

```json
{
  "n":       {"db": 3657, "val": 312, "test": 570},
  "classes": ["ADH", "DCIS", "FEA", "IC", "N", "PB", "UDH"],
  "chance":  {"random_retrieval_acc@1": 0.14..., "majority_class_acc@1": 0.14...},
  "val":     {"metrics": {"acc@1":…, "acc@3":…, "mvacc@3":…, "acc@5":…, "mvacc@5":…},
              "per_class_acc@1": {"ADH": 0.0…, …}},
  "test":    {"metrics": {…same keys…}, "per_class_acc@1": {…}}
}
```

## 4. Steps

### Step 1 — add section-preserving write logic

Replace the bare `OUT.write_text(...)` with a merge. Add these two helpers **above** `main()`
in `validation/make_results.py`:

```python
def split_sections(text):
    """Parse a RESULTS.md into (preamble_lines, [(title, [body_lines]), ...]) by `## ` heading."""
    preamble, sections, cur = [], [], None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = (line[3:].strip(), [line])
            sections.append(cur)
        elif cur is None:
            preamble.append(line)
        else:
            cur[1].append(line)
    return preamble, sections


def merge(old_text, preamble, generated):
    """RESULTS.md is update-only: regenerate what we can, carry everything else through.

    `generated` is an ordered [(title, [lines])]. Any section present in `old_text` but not in
    `generated` (a hand-written analysis section, or one whose result JSON is absent on this
    machine) is preserved verbatim, in its original position. Sections we generate for the
    first time are appended in generator order.
    """
    gen = dict(generated)
    _, old_sections = split_sections(old_text)
    out, emitted = list(preamble), set()

    for title, body in old_sections:              # keep the old file's section order
        if title in gen:
            out += gen[title]
        else:
            print(f"[make_results] preserved (not generated): {title}")
            out += body
        emitted.add(title)

    for title, body in generated:                 # new sections we've never written before
        if title not in emitted:
            out += body

    return "\n".join(out).rstrip() + "\n"
```

### Step 2 — restructure `main()` to build `(title, lines)` pairs

`main()` currently appends into one flat `lines` list. Change it to append into a `generated`
list of `(title, lines)` tuples. The **title string must exactly match** the heading text that
appears in `RESULTS.md` today (without the leading `## `), or the merge will duplicate the
section instead of replacing it. Use these exact titles:

| JSON | Exact section title |
|---|---|
| `smoke_slide_encoding.json` | `Smoke test — slide encoding` |
| `classification_camelyon16.json` | `Classification — CAMELYON16 binary (tumor vs normal)` |
| `classification_tcga_ot.json` | `Classification — TCGA-OT (46-class, full WSIs)` |
| `retrieval_tcga_ot.json` | `Retrieval — TCGA-OT slide retrieval (patient-disjoint)` |
| Tier-1 group | `Retrieval Tier-1 — training-free post-processing (negative result)` |
| `subtasks_tcga_ot.json` | `Sub-tasks — harder patient-disjoint TCGA-OT sub-typing` |
| `retrieval_bracs.json` | `Retrieval — BRACS ROI (patient-disjoint baseline)` (new, step 3) |

> ⚠️ Those are **em-dashes (`—`, U+2014)**, not hyphens. Copy them from the table above or from
> `RESULTS.md`; a hyphen here silently breaks the match and duplicates the section.

The body-generating code inside each `if <json>:` block does not change — only where its lines
are collected. Keep the preamble as-is:

```python
    preamble = ["# TITAN Validation Results", "",
                "Reference numbers from the TITAN Nature Medicine paper / repo README.", ""]
    generated = []
```

and at the end of `main()`:

```python
    old = OUT.read_text() if OUT.exists() else ""
    OUT.write_text(merge(old, preamble, generated))
    print(f"[make_results] wrote {OUT}")
```

Note the final `print("\n".join(lines))` (dumping the whole file to stdout) can go; keep the
`[make_results] wrote` line and the `preserved (not generated)` lines.

### Step 3 — add the BRACS baseline section

The BRACS raw-cosine baseline has a result JSON but no section anywhere. Add a generator for
it, placed after the Sub-tasks block, using the exact title from the table above:

```python
    bracs = load("retrieval_bracs.json")
    if bracs:
        te, ch = bracs["test"], bracs["chance"]
        body = [f"## Retrieval — BRACS ROI (patient-disjoint baseline)", "",
                f"Frozen TITAN, raw cosine. DB(train)={bracs['n']['db']}  "
                f"val_q={bracs['n']['val']}  test_q={bracs['n']['test']}  "
                f"{len(bracs['classes'])} classes.",
                "",
                "| Metric | Test | Chance (random retrieval) | Chance (majority class) |",
                "|---|---|---|---|",
                f"| Acc@1 | {fmt(te['metrics']['acc@1'])} | "
                f"{fmt(ch['random_retrieval_acc@1'])} | {fmt(ch['majority_class_acc@1'])} |",
                f"| Acc@3 | {fmt(te['metrics']['acc@3'])} | — | — |",
                f"| MVAcc@3 | {fmt(te['metrics']['mvacc@3'])} | — | — |",
                "",
                "Per-class Acc@1: " + ", ".join(
                    f"{c}={v:.3f}" for c, v in te["per_class_acc@1"].items()),
                ""]
        generated.append(("Retrieval — BRACS ROI (patient-disjoint baseline)", body))
```

Because this title is new, `merge()` appends it at the end on first run — **after** the
preserved Tier-3 section. That is acceptable. If you want it before Tier-3, move it manually in
`RESULTS.md` once; `merge()` respects the old file's order from then on.

### Step 4 — do NOT add a Tier-3 generator

Leave `## Retrieval Tier-3 …` unmanaged. Its value is the hand-written analysis, which no JSON
can reproduce. `merge()` now preserves it, and prints
`[make_results] preserved (not generated): Retrieval Tier-3 — …` so it is visible that it is
being carried, not regenerated.

## 5. Verification (run each; all must pass)

Run from the repo root, with the project venv:

```bash
# 1. Baseline: record the current file so you can prove nothing was lost.
cp validation/RESULTS.md /tmp/RESULTS.before.md
grep -c '^## ' /tmp/RESULTS.before.md          # expect: 6
wc -l < /tmp/RESULTS.before.md                 # expect: 111

# 2. Regenerate.
cd validation && ../.venv/bin/python make_results.py && cd ..

# 3. The Tier-3 section MUST still be there, byte-for-byte.
grep -q '^## Retrieval Tier-3' validation/RESULTS.md && echo "TIER-3 PRESERVED"
diff <(sed -n '/^## Retrieval Tier-3/,$p' /tmp/RESULTS.before.md) \
     <(sed -n '/^## Retrieval Tier-3/,$p' validation/RESULTS.md) && echo "TIER-3 IDENTICAL"

# 4. The BRACS baseline section is new and present.
grep -q '^## Retrieval — BRACS ROI' validation/RESULTS.md && echo "BRACS SECTION ADDED"

# 5. Section count went UP, never down.
test "$(grep -c '^## ' validation/RESULTS.md)" -ge 6 && echo "NO SECTIONS LOST"

# 6. Idempotence: running twice changes nothing.
cp validation/RESULTS.md /tmp/RESULTS.once.md
cd validation && ../.venv/bin/python make_results.py && cd ..
diff /tmp/RESULTS.once.md validation/RESULTS.md && echo "IDEMPOTENT"
```

Expected stdout from step 2 includes:
```
[make_results] preserved (not generated): Retrieval Tier-3 — LoRA fine-tuning of the TITAN slide encoder (BRACS ROIs)
[make_results] wrote /home/user01/TITAN/validation/RESULTS.md
```

`make_results.py` reads only local JSON — **no network, no GPU, no HF token required.**

## 6. Done criteria (machine-checkable)

- [ ] `grep -q '^## Retrieval Tier-3' validation/RESULTS.md` exits 0 **after** running `make_results.py`.
- [ ] The Tier-3 section body is byte-identical to the pre-change version (`diff` in step 3 is empty).
- [ ] `grep -c '^## ' validation/RESULTS.md` returns ≥ 7 (the original 6 + BRACS).
- [ ] Running `make_results.py` twice in a row is a no-op (`diff` in step 6 is empty).
- [ ] `bash validation/run_all.sh` no longer destroys anything — but **do not run it**, it needs
      a GPU, an HF token and ~1 h. Step 6's idempotence check covers the same code path.

## 7. Scope

**In scope:** `validation/make_results.py` only.

**Out of scope — do not touch:**
- `validation/RESULTS.md` — do not hand-edit it. It must be *produced* by the fixed script. The
  only acceptable diff to it is the newly appended BRACS section.
- `validation/run_all.sh` — it becomes safe once `make_results.py` is fixed.
- Any `validation/results/*.json` — these are experimental records; never edit or regenerate them.
- Any experiment script (`retrieval_*.py`, `classification_*.py`, `finetune_*.py`) — this plan
  changes reporting only, never a number.

## 8. Test plan

Add `validation/tests/test_make_results.py` (create the dir; see plan 003 for the pytest
harness and `conftest.py` — if plan 003 has not landed yet, this test can still be run with
`python -m pytest validation/tests/test_make_results.py` once `pytest` is installed):

```python
"""make_results.py must never delete a section it cannot regenerate."""
import make_results


def test_merge_preserves_unknown_sections():
    old = "# T\n\nintro\n\n## Alpha\nold alpha\n\n## Handwritten\nprecious analysis\n"
    generated = [("Alpha", ["## Alpha", "new alpha", ""])]
    out = make_results.merge(old, ["# T", "", "intro", ""], generated)
    assert "## Handwritten" in out
    assert "precious analysis" in out          # the whole point of this plan
    assert "new alpha" in out                  # regenerated section was replaced
    assert "old alpha" not in out


def test_merge_keeps_old_order_and_appends_new():
    old = "# T\n\n## B\nb\n\n## A\na\n"
    generated = [("A", ["## A", "a2", ""]), ("C", ["## C", "c", ""])]
    out = make_results.merge(old, ["# T", ""], generated)
    assert out.index("## B") < out.index("## A") < out.index("## C")


def test_merge_on_empty_file_is_just_the_generated_sections():
    out = make_results.merge("", ["# T", ""], [("A", ["## A", "a", ""])])
    assert out.startswith("# T")
    assert "## A" in out
```

Run: `.venv/bin/python -m pytest validation/tests/test_make_results.py -q` → 3 passed.

## 9. Maintenance note

The invariant to defend in review from now on is: **`make_results.py` may add and update
sections in `RESULTS.md`; it may never remove one.** Any future PR that adds a
`lines += [...]`-style section must register a `(title, lines)` pair whose title exactly matches
the heading it emits, or `merge()` will treat the new heading as unknown and duplicate it.

The em-dash in the section titles is a real trap. If you see a section appearing twice in
`RESULTS.md`, the first thing to check is `—` (U+2014) vs `-` (U+002D) in the title string.

## 10. Escape hatches — STOP and report back if:

- `validation/RESULTS.md` at `HEAD` does **not** contain `## Retrieval Tier-3` (someone already
  regenerated and committed the truncated file — the content must be recovered from
  `git show 9d56a69:validation/RESULTS.md` *before* you do anything else).
- The idempotence check (step 6) fails: two consecutive runs produce different files. Do not
  "fix" this by loosening the diff — it means `merge()` is mis-parsing, and a mis-parse can
  still drop content.
- Any `validation/results/*.json` file has a shape that differs from what §3 documents. Report
  the actual shape; do not guess at keys.
