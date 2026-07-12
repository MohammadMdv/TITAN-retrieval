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


def main():
    preamble = ["# TITAN Validation Results", "",
                "Reference numbers from the TITAN Nature Medicine paper / repo README.", ""]
    generated = []

    smoke = load("smoke_slide_encoding.json")
    if smoke:
        lines = ["## Smoke test — slide encoding",
                  f"- sample: `{smoke['sample']}`  embedding shape: {smoke['embedding_shape']}  "
                  f"patch_size_lv0: {smoke['patch_size_lv0']}  finite: {smoke['finite']}", ""]
        generated.append(("Smoke test — slide encoding", lines))

    cam = load("classification_camelyon16.json")
    if cam:
        lines = ["## Classification — CAMELYON16 binary (tumor vs normal)",
                  f"- {cam['n_slides']} slides ({cam['n_normal']} normal / {cam['n_tumor']} tumor). "
                  f"_Caveat: 512px@40x features (CONCHv1.5 expects ~20x); not a TITAN-designed task._",
                  "",
                  "| Setting | Balanced acc | AUROC |",
                  "|---|---|---|",
                  f"| Zero-shot | {fmt(cam['zero_shot'].get('bacc'))} | {fmt(cam['zero_shot'].get('auroc'))} |",
                  f"| Linear probe | {fmt(cam['linear_probe'].get('bacc'))} | {fmt(cam['linear_probe'].get('auroc'))} |",
                  f"- LP bootstrap: {cam.get('linear_probe_bootstrap')}", ""]
        generated.append(("Classification — CAMELYON16 binary (tumor vs normal)", lines))

    ot = load("classification_tcga_ot.json")
    if ot:
        lp, zs = ot["linear_probe"], ot.get("zero_shot")
        lines = ["## Classification — TCGA-OT (46-class, full WSIs)",
                  "| Setting | Balanced acc | Reference | AUROC | Kappa |",
                  "|---|---|---|---|---|",
                  f"| Linear probe | {fmt(lp.get('bacc'))} | **0.704** | {fmt(lp.get('auroc'))} | {fmt(lp.get('kappa'))} |"]
        if zs:
            lines.append(f"| Zero-shot | {fmt(zs.get('bacc'))} | — | {fmt(zs.get('auroc'))} | {fmt(zs.get('kappa'))} |")
        lines += [f"- LP bootstrap: {ot.get('linear_probe_bootstrap')}",
                  f"- ZS bootstrap: {ot.get('zero_shot_bootstrap')}", ""]
        generated.append(("Classification — TCGA-OT (46-class, full WSIs)", lines))

    ret = load("retrieval_tcga_ot.json")
    if ret:
        lines = ["## Retrieval — TCGA-OT slide retrieval (patient-disjoint)",
                  f"- DB={ret['db_size']}  queries={ret['n_queries']}  "
                  f"leaking patients dropped={ret['n_leaking_dropped']}  "
                  f"disjoint asserted={ret['patient_disjoint_asserted']}",
                  "",
                  "| Metric | Value | Reference |",
                  "|---|---|---|",
                  f"| Acc@3 | {fmt(ret['acc@3'])} | **0.880** |",
                  f"| MVAcc@3 | {fmt(ret['mvacc@3'])} | **0.807** |", ""]
        generated.append(("Retrieval — TCGA-OT slide retrieval (patient-disjoint)", lines))

    # ---- Tier-1: training-free retrieval post-processing on frozen embeddings ----
    tier1 = [("Whitening / PCA", "retrieval_tcga_ot_whitening.json"),
             ("k-reciprocal re-ranking", "retrieval_tcga_ot_kreciprocal.json"),
             ("Query expansion (αQE)", "retrieval_tcga_ot_query_expansion.json"),
             ("Database-side augmentation (DBA)", "retrieval_tcga_ot_query_expansion.json")]
    loaded = {n: load(f) for _, f in tier1 for n in [f]}
    if any(loaded.values()):
        lines = ["## Retrieval Tier-1 — training-free post-processing (negative result)",
                  "",
                  "Three standard image-retrieval techniques applied to the **frozen** TITAN "
                  "embeddings. Nothing is trained; only the search procedure changes. "
                  "Hyperparameters are selected on **val** queries and reported once on **test** "
                  "queries, against a fixed patient-disjoint database.",
                  "",
                  "**None of them beat the raw-cosine baseline.** Selected configs and their test "
                  "deltas (selection metric Acc@3):", ""]

        rows = []
        w = loaded.get("retrieval_tcga_ot_whitening.json")
        if w:
            rows.append(("Whitening", w["selected"]["config"], w["baseline"]["test"], w["test"]))
        k = loaded.get("retrieval_tcga_ot_kreciprocal.json")
        if k:
            rows.append(("k-reciprocal", k["selected"]["config"], k["baseline"]["test"], k["test"]))
        q = loaded.get("retrieval_tcga_ot_query_expansion.json")
        if q:
            rows.append(("αQE", q["qe"]["selected"]["config"], q["baseline"]["test"], q["qe"]["test"]))
            rows.append(("DBA", q["dba"]["selected"]["config"], q["baseline"]["test"], q["dba"]["test"]))

        lines += ["| Technique | Selected config (on val) | Acc@3 | ΔAcc@3 | MVAcc@3 | ΔMVAcc@3 |",
                  "|---|---|---|---|---|---|"]
        base = rows[0][2] if rows else None
        if base:
            lines.append(f"| _baseline (raw cosine)_ | — | {fmt(base['acc@3'])} | — | "
                         f"{fmt(base['mvacc@3'])} | — |")
        for name, cfg, b, t in rows:
            lines.append(f"| {name} | `{cfg}` | {fmt(t['acc@3'])} | {delta(t['acc@3'], b['acc@3'])} | "
                         f"{fmt(t['mvacc@3'])} | {delta(t['mvacc@3'], b['mvacc@3'])} |")
        lines += ["",
                  "Notes:",
                  "- k-reciprocal's val-optimal setting is **λ=1.0**, which by construction *is* the "
                  "plain cosine ranking — the sweep chose \"do not re-rank at all\". (λ=1.0 "
                  "reproducing the baseline exactly also serves as the implementation's sanity check.)",
                  "- Whitening's `pca-384` row is **not** a win: it *lost* to the baseline on val "
                  "(−0.0012) and was only selected as the sweep's argmax. Its +0.0007 test Acc@3 is "
                  "a single query out of 1348 — noise, not effect.",
                  "- Full whitening (`pcaw-*`, `zca`) actively hurts Acc@3, implying TITAN's "
                  "high-variance directions carry discriminative signal rather than nuisance "
                  "variation — the usual motivation for whitening does not apply here.",
                  "- Every technique showed a small **MVAcc@3 gain on val** (+0.010 to +0.013), "
                  "which looked like a real trade (more label-homogeneous top-3 at the cost of "
                  "Acc@3). Re-selecting on MVAcc@3 and re-testing showed those gains **do not "
                  "transfer to test** — all four flip negative (whitening −0.0074, k-reciprocal "
                  "−0.0015, αQE −0.0067, DBA −0.0037). They were val-set noise, not signal. "
                  "Those runs are saved alongside as `*_sel-mvacc3.json`.", ""]
        generated.append(("Retrieval Tier-1 — training-free post-processing (negative result)", lines))

    sub = load("subtasks_tcga_ot.json")
    if sub:
        lines = ["## Sub-tasks — harder patient-disjoint TCGA-OT sub-typing", "",
                  "| Task | #cls | test n | LP bacc | Ret Acc@1 | MVAcc@3 |", "|---|---|---|---|---|---|"]
        for name, r in sub.items():
            lines.append(f"| {name} | {len(r['classes'])} | {r['n']['test']} | "
                         f"{fmt(r['linear_probe'].get('bacc'))} | {fmt(r['retrieval']['acc@1'])} | "
                         f"{fmt(r['retrieval']['mvacc@3'])} |")
        lines.append("")
        generated.append(("Sub-tasks — harder patient-disjoint TCGA-OT sub-typing", lines))

    bracs = load("retrieval_bracs.json")
    if bracs:
        te, ch = bracs["test"], bracs["chance"]
        body = ["## Retrieval — BRACS ROI (patient-disjoint baseline)", "",
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

    old = OUT.read_text() if OUT.exists() else ""
    OUT.write_text(merge(old, preamble, generated))
    print(f"[make_results] wrote {OUT}")


if __name__ == "__main__":
    main()
