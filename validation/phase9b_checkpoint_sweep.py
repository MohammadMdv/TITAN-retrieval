"""Phase 9b: does the checkpoint phase9 picked (lowest val SupCon loss) actually track the
best downstream metric?

Context: the first real phase8/phase9 run selected epoch 5/50 (val SupCon loss rose
monotonically after that -- clear overfitting), and the resulting AFTER numbers were mixed:
LP bacc +0.43pp, but Acc@3 -1.34pp and MVAcc@3 -0.59pp vs the raw-embedding baseline. That
checkpoint had only ~320 gradient steps -- barely trained. Two live questions this answers:
  1. Is epoch 5 actually near-optimal for the downstream task too, or does the SupCon val
     loss (a proxy) diverge from what LP/retrieval actually want?
  2. Does ANY epoch in the run beat the raw-embedding baseline on retrieval, not just LP?

Evaluates LP + retrieval (identical phase3/phase4 protocol, via phase9's own functions) at a
sparse set of the per-epoch checkpoints phase8 now saves, prints a table of
epoch -> val_supcon_loss -> LP bacc -> Acc@3 -> MVAcc@3, alongside the raw-embedding baseline.

Requires phase8_finetune_supcon.py to have been run (reads ssl_checkpoints/epoch_NNN.pt and
phase8_finetune_supcon.json for the val-loss column).

Env: SWEEP_EPOCHS="1,3,5,7,10,15,20,30,40,50" (comma-separated epoch numbers, 1-indexed;
defaults to that set -- dense early where the val loss was still improving, sparse later).
Each swept epoch costs one full LP C-sweep + retrieval pass (~30-60s), so the 10-epoch
default takes roughly 5-10 minutes total.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import json

import numpy as np
import torch
import yaml

from common import get_device, save_results, RESULTS_DIR, CONFIG_TCGA_OT
from ssl_common import load_embeddings, build_split, ProjHead, PROJ_DIM
from phase9_eval_before_after import evaluate, project, REF

CKPT_DIR = RESULTS_DIR / "ssl_checkpoints"
HISTORY_JSON = RESULTS_DIR / "phase8_finetune_supcon.json"

DEFAULT_EPOCHS = "1,3,5,7,10,15,20,30,40,50"


def main():
    sys.stdout.reconfigure(line_buffering=True)  # visible progress even when piped/redirected
    with open(CONFIG_TCGA_OT) as f:
        label_dict = yaml.safe_load(f)["label_dict"]

    if not CKPT_DIR.exists():
        print(f"[phase9b] {CKPT_DIR} not found -- run phase8_finetune_supcon.py first.")
        return 1

    device = get_device()
    print("[phase9b] loading cached embeddings + TCGA-OT splits ...")
    emb_df = load_embeddings()
    Xtr, ytr, ctr, idtr = build_split(emb_df, "train")
    Xva, yva, cva, idva = build_split(emb_df, "val")
    Xte, yte, cte, idte = build_split(emb_df, "test")

    val_loss_by_epoch = {}
    if HISTORY_JSON.exists():
        hist = json.loads(HISTORY_JSON.read_text())
        val_loss_by_epoch = {h["epoch"] + 1: h["val_loss"] for h in hist.get("history", [])}

    epochs = [int(e) for e in os.environ.get("SWEEP_EPOCHS", DEFAULT_EPOCHS).split(",")]
    available = sorted(int(p.stem.split("_")[1]) for p in CKPT_DIR.glob("epoch_*.pt"))
    epochs = [e for e in epochs if e in available]
    missing = [e for e in epochs if e not in available]
    if missing:
        print(f"[phase9b] skipping epochs not on disk: {missing} (available: {available[0]}-{available[-1]})")
    print(f"[phase9b] sweeping epochs: {epochs}")

    print(f"\n[phase9b] === BASELINE (raw TITAN embeddings; ref bacc={REF['lp_bacc']} "
          f"acc@3={REF['acc@3']} mvacc@3={REF['mvacc@3']}) ===")
    lp_base, rt_base = evaluate(Xtr, ytr, ctr, Xva, yva, cva, Xte, yte, cte, label_dict)
    print(f"[phase9b] BASELINE: LP bacc={lp_base['bacc']:.4f}  Acc@3={rt_base['acc@3']:.4f}  "
          f"MVAcc@3={rt_base['mvacc@3']:.4f}")

    rows = [{"epoch": 0, "val_supcon_loss": None, "lp_bacc": lp_base["bacc"],
             "acc@3": rt_base["acc@3"], "mvacc@3": rt_base["mvacc@3"], "label": "baseline (no head)"}]

    for ep in epochs:
        ckpt = CKPT_DIR / f"epoch_{ep:03d}.pt"
        head = ProjHead(Xtr.shape[1], PROJ_DIM).to(device)
        head.load_state_dict(torch.load(ckpt, map_location=device))
        Ztr, Zva, Zte = project(head, Xtr, device), project(head, Xva, device), project(head, Xte, device)
        lp, rt = evaluate(Ztr, ytr, ctr, Zva, yva, cva, Zte, yte, cte, label_dict)
        vl = val_loss_by_epoch.get(ep)
        vl_str = f"{vl:.4f}" if vl is not None else "n/a"
        print(f"[phase9b] epoch {ep:3d}: val_supcon_loss={vl_str}  LP bacc={lp['bacc']:.4f}  "
              f"Acc@3={rt['acc@3']:.4f}  MVAcc@3={rt['mvacc@3']:.4f}", flush=True)
        rows.append({"epoch": ep, "val_supcon_loss": vl, "lp_bacc": lp["bacc"],
                     "acc@3": rt["acc@3"], "mvacc@3": rt["mvacc@3"], "label": f"epoch_{ep:03d}"})
        save_results("phase9b_checkpoint_sweep.json", {"rows": rows})  # incremental

    print("\n" + "=" * 78)
    print(f"[phase9b] SWEEP TABLE  (baseline: LP={lp_base['bacc']:.4f} Acc@3={rt_base['acc@3']:.4f} "
          f"MVAcc@3={rt_base['mvacc@3']:.4f})")
    print(f"  {'epoch':>6s}  {'val_loss':>9s}  {'LP_bacc':>8s}  {'dLP':>7s}  "
          f"{'Acc@3':>7s}  {'dAcc@3':>8s}  {'MVAcc@3':>8s}  {'dMVAcc@3':>9s}")
    for r in rows:
        vl_str = f"{r['val_supcon_loss']:.4f}" if r["val_supcon_loss"] is not None else "-"
        dlp = r["lp_bacc"] - lp_base["bacc"]
        dacc = r["acc@3"] - rt_base["acc@3"]
        dmv = r["mvacc@3"] - rt_base["mvacc@3"]
        print(f"  {r['label']:>12s}  {vl_str:>9s}  {r['lp_bacc']:>8.4f}  {dlp:>+7.4f}  "
              f"{r['acc@3']:>7.4f}  {dacc:>+8.4f}  {r['mvacc@3']:>8.4f}  {dmv:>+9.4f}")

    best_retrieval = max(rows[1:], key=lambda r: r["acc@3"], default=None)
    best_lp = max(rows[1:], key=lambda r: r["lp_bacc"], default=None)
    if best_retrieval:
        print(f"\n  Best Acc@3 across swept epochs: epoch {best_retrieval['epoch']} "
              f"(Acc@3={best_retrieval['acc@3']:.4f}, baseline={rt_base['acc@3']:.4f})")
        print(f"  Best LP bacc across swept epochs: epoch {best_lp['epoch']} "
              f"(bacc={best_lp['lp_bacc']:.4f}, baseline={lp_base['bacc']:.4f})")

    save_results("phase9b_checkpoint_sweep.json", {"rows": rows, "baseline": {
        "lp_bacc": lp_base["bacc"], "acc@3": rt_base["acc@3"], "mvacc@3": rt_base["mvacc@3"]}})
    print("[phase9b] OK")


if __name__ == "__main__":
    sys.exit(main())
