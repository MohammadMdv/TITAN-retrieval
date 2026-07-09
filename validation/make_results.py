"""Aggregate per-phase JSON results into RESULTS.md (compared to paper reference numbers)."""
import json
from pathlib import Path
from common import RESULTS_DIR

OUT = Path(__file__).resolve().parent / "RESULTS.md"


def load(name):
    p = RESULTS_DIR / name
    return json.load(open(p)) if p.exists() else None


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    lines = ["# TITAN Validation Results", ""]
    lines.append("Reference numbers from the TITAN Nature Medicine paper / repo README.")
    lines.append("")

    p1 = load("phase1_smoke.json")
    if p1:
        lines += ["## Phase 1 — slide-encoding smoke test",
                  f"- sample: `{p1['sample']}`  embedding shape: {p1['embedding_shape']}  "
                  f"patch_size_lv0: {p1['patch_size_lv0']}  finite: {p1['finite']}", ""]

    p2 = load("phase2_camelyon.json")
    if p2:
        lines += ["## Phase 2 — CAMELYON16 binary (tumor vs normal)",
                  f"- {p2['n_slides']} slides ({p2['n_normal']} normal / {p2['n_tumor']} tumor). "
                  f"_Caveat: 512px@40x features (CONCHv1.5 expects ~20x); not a TITAN-designed task._",
                  "",
                  "| Setting | Balanced acc | AUROC |",
                  "|---|---|---|",
                  f"| Zero-shot | {fmt(p2['zero_shot'].get('bacc'))} | {fmt(p2['zero_shot'].get('auroc'))} |",
                  f"| Linear probe | {fmt(p2['linear_probe'].get('bacc'))} | {fmt(p2['linear_probe'].get('auroc'))} |",
                  f"- LP bootstrap: {p2.get('linear_probe_bootstrap')}", ""]

    p3 = load("phase3_tcga_ot.json")
    if p3:
        lp = p3["linear_probe"]; zs = p3["zero_shot"]
        lines += ["## Phase 3 — TCGA-OT (46-class, full WSIs)",
                  "| Setting | Balanced acc | Reference | AUROC | Kappa |",
                  "|---|---|---|---|---|",
                  f"| Linear probe | {fmt(lp.get('bacc'))} | **0.704** | {fmt(lp.get('auroc'))} | {fmt(lp.get('kappa'))} |",
                  f"| Zero-shot | {fmt(zs.get('bacc'))} | — | {fmt(zs.get('auroc'))} | {fmt(zs.get('kappa'))} |",
                  f"- LP bootstrap: {p3.get('linear_probe_bootstrap')}",
                  f"- ZS bootstrap: {p3.get('zero_shot_bootstrap')}", ""]

    p4 = load("phase4_retrieval.json")
    if p4:
        lines += ["## Phase 4 — TCGA-OT slide retrieval (patient-disjoint)",
                  f"- DB={p4['db_size']}  queries={p4['n_queries']}  "
                  f"leaking patients dropped={p4['n_leaking_dropped']}  "
                  f"disjoint asserted={p4['patient_disjoint_asserted']}",
                  "",
                  "| Metric | Value | Reference |",
                  "|---|---|---|",
                  f"| Acc@3 | {fmt(p4['acc@3'])} | **0.880** |",
                  f"| MVAcc@3 | {fmt(p4['mvacc@3'])} | **0.807** |", ""]

    p8 = load("phase8_finetune_supcon.json")
    p9b = load("phase9b_checkpoint_sweep.json")
    if p8 or p9b:
        lines += ["## Phase 8/9/9b — SupCon projection-head fine-tuning on frozen TITAN "
                   "embeddings (negative result)",
                  "TITAN stays frozen throughout; only a small projection head (768→768→768) "
                  "is trained with SupCon (Khosla et al. 2020) on the same OncoTreeCode labels the "
                  "linear probe uses. Question: does a label-aware nonlinear projection of the frozen "
                  "embedding beat the raw embedding on TCGA-OT retrieval/LP? **Answer: no** — val "
                  "SupCon loss overfits almost immediately, and retrieval degrades monotonically with "
                  "more training while LP bacc stays roughly flat.", ""]
        if p8:
            lines += [f"- Training: {p8['n_epochs']} epochs, P={p8['p_classes']}×K={p8['k_samples']} "
                       f"({p8['p_classes']*p8['k_samples']} per batch), best val SupCon loss at "
                       f"epoch {p8['best_epoch']+1}/{p8['n_epochs']}.", ""]
        if p9b:
            base = p9b["baseline"]
            lines += ["| Epoch | val SupCon loss | LP bacc | ΔLP | Acc@3 | ΔAcc@3 | MVAcc@3 | ΔMVAcc@3 |",
                      "|---|---|---|---|---|---|---|---|",
                      f"| baseline (no head) | — | {fmt(base['lp_bacc'])} | — | "
                      f"{fmt(base['acc@3'])} | — | {fmt(base['mvacc@3'])} | — |"]
            for r in p9b["rows"]:
                if r["epoch"] == 0:
                    continue
                vl = fmt(r["val_supcon_loss"]) if r["val_supcon_loss"] is not None else "-"
                lines.append(
                    f"| {r['epoch']} | {vl} | {fmt(r['lp_bacc'])} | {r['lp_bacc']-base['lp_bacc']:+.4f} | "
                    f"{fmt(r['acc@3'])} | {r['acc@3']-base['acc@3']:+.4f} | "
                    f"{fmt(r['mvacc@3'])} | {r['mvacc@3']-base['mvacc@3']:+.4f} |")
            lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"[make_results] wrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
