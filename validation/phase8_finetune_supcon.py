"""Phase 8: SupCon fine-tuning of a small projection head on frozen TITAN embeddings
(TCGA-OT, 46-class OncoTreeCode, patient-disjoint train/val).

TITAN is never touched -- this trains only the head (768 -> 768 -> 256), on the precomputed
TCGA_TITAN_features.pkl embeddings already used by phase3/phase4. No TITAN forward pass, so
none of the ALiBi re-encode cost from phase6 applies here; phase7 already showed the head+
SupCon-loss step itself is ~free on VRAM (<0.05GB @ batch=128). Expect this to finish in well
under a couple of minutes.

Batch sampling: PK sampler (P classes x K slides/class per batch), not plain random shuffling.
46 classes range 34-798 slides in train; random batches would leave most of the 46 classes'
anchors with zero in-batch positives most steps. PK sampling guarantees every anchor has
exactly K-1 positives every step, which is what SupCon needs to produce a useful gradient
for the rare classes too.

Model selection: keep the checkpoint with the lowest SupCon loss on the untouched val split
(same val-based selection philosophy as the linear probe's C sweep). Test split is never
touched here at all -- phase9 is the first time it's used.

Outputs (scratch, RESULTS_DIR): ssl_head_best.pt, ssl_head_final.pt, phase8_finetune_supcon.json,
and a per-epoch snapshot in ssl_checkpoints/epoch_NNN.pt (cheap: ~3MB x N_EPOCHS) so
phase9b_checkpoint_sweep.py can evaluate downstream metrics at multiple epochs without
re-training -- useful for checking whether "best val SupCon loss" actually tracks "best
downstream LP/retrieval", since a supervised proxy loss and the true task metric can diverge
early in training.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import random
import time

import numpy as np
import torch

from common import get_device, save_results, RESULTS_DIR
from ssl_common import load_embeddings, build_split, ProjHead, supcon_loss, PROJ_DIM

CKPT_BEST = RESULTS_DIR / "ssl_head_best.pt"
CKPT_FINAL = RESULTS_DIR / "ssl_head_final.pt"
CKPT_DIR = RESULTS_DIR / "ssl_checkpoints"   # per-epoch snapshots, for phase9b's sweep

P_CLASSES = int(os.environ.get("P_CLASSES", 16))   # distinct classes sampled per batch
K_SAMPLES = int(os.environ.get("K_SAMPLES", 8))    # slides per chosen class per batch
N_EPOCHS = int(os.environ.get("N_EPOCHS", 50))
LR = float(os.environ.get("LR", 1e-3))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.07))
SEED = 0


def pk_batch(X, y, class_to_idx, classes, P, K, rng, device):
    chosen = rng.sample(classes, min(P, len(classes)))
    idx = []
    for c in chosen:
        pool = class_to_idx[c]
        if len(pool) >= K:
            idx += rng.sample(pool, K)
        else:  # small class: sample with replacement rather than skip it entirely
            idx += [rng.choice(pool) for _ in range(K)]
    idx = np.array(idx)
    return torch.from_numpy(X[idx]).to(device), torch.from_numpy(y[idx]).to(device)


@torch.no_grad()
def val_loss(head, Xv, yv, device, temperature):
    head.eval()
    z = head(torch.from_numpy(Xv).to(device))
    loss = supcon_loss(z, torch.from_numpy(yv).to(device), temperature)
    head.train()
    return float(loss.item())


def main():
    sys.stdout.reconfigure(line_buffering=True)  # visible progress even when piped/redirected
    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("CUDA not available. Phase 7 showed this workload's VRAM cost is "
                           "trivial, so this should never be a resource problem -- check "
                           "`nvidia-smi` for a driver issue before retrying.")
    torch.manual_seed(SEED)

    print("[phase8] loading cached TCGA_TITAN_features.pkl + TCGA-OT splits ...")
    emb_df = load_embeddings()
    Xtr, ytr_raw, ctr, idtr = build_split(emb_df, "train")
    Xva, yva_raw, cva, idva = build_split(emb_df, "val")
    print(f"[phase8] train={len(ytr_raw)}  val={len(yva_raw)}")

    classes_sorted = sorted(set(ytr_raw) | set(yva_raw))
    lab2idx = {c: i for i, c in enumerate(classes_sorted)}
    ytr = np.array([lab2idx[c] for c in ytr_raw], dtype=np.int64)
    yva = np.array([lab2idx[c] for c in yva_raw], dtype=np.int64)

    class_to_idx = {}
    for i, c in enumerate(ytr):
        class_to_idx.setdefault(int(c), []).append(i)
    classes = list(class_to_idx.keys())
    min_class = min(len(v) for v in class_to_idx.values())
    print(f"[phase8] {len(classes)} classes in train, min class size={min_class}, "
          f"PK sampler P={P_CLASSES} K={K_SAMPLES} (batch={P_CLASSES * K_SAMPLES})")

    CKPT_DIR.mkdir(exist_ok=True)
    head = ProjHead(Xtr.shape[1], PROJ_DIM).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    steps_per_epoch = max(1, len(ytr) // (P_CLASSES * K_SAMPLES))
    rng = random.Random(SEED)
    best_val, best_epoch = float("inf"), -1
    history = []
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        ep_losses = []
        for _ in range(steps_per_epoch):
            xb, yb = pk_batch(Xtr, ytr, class_to_idx, classes, P_CLASSES, K_SAMPLES, rng, device)
            z = head(xb)
            loss = supcon_loss(z, yb, TEMPERATURE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        sched.step()
        vl = val_loss(head, Xva, yva, device, TEMPERATURE)
        history.append({"epoch": epoch, "train_loss": float(np.mean(ep_losses)), "val_loss": vl})
        torch.save(head.state_dict(), CKPT_DIR / f"epoch_{epoch + 1:03d}.pt")
        if vl < best_val:
            best_val, best_epoch = vl, epoch
            torch.save(head.state_dict(), CKPT_BEST)
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch + 1}/{N_EPOCHS}: train_loss={np.mean(ep_losses):.4f} "
                  f"val_loss={vl:.4f} (best={best_val:.4f} @ep{best_epoch + 1})", flush=True)

    torch.save(head.state_dict(), CKPT_FINAL)
    dt = time.time() - t0
    print(f"[phase8] done in {dt:.1f}s. best val_loss={best_val:.4f} @ epoch {best_epoch + 1}")
    print(f"[phase8] checkpoints: {CKPT_BEST}  {CKPT_FINAL}")
    print(f"[phase8] per-epoch snapshots: {CKPT_DIR}/epoch_NNN.pt ({N_EPOCHS} files)")

    save_results("phase8_finetune_supcon.json", {
        "p_classes": P_CLASSES, "k_samples": K_SAMPLES, "n_epochs": N_EPOCHS, "lr": LR,
        "temperature": TEMPERATURE, "steps_per_epoch": steps_per_epoch,
        "best_val_loss": best_val, "best_epoch": best_epoch, "train_time_s": dt,
        "history": history,
    })
    print("[phase8] OK -- next: run phase9_eval_before_after.py, or phase9b_checkpoint_sweep.py "
          "to compare multiple epochs")


if __name__ == "__main__":
    sys.exit(main())
