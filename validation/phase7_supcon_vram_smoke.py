"""Phase 7: SupCon fine-tuning feasibility smoke test (VRAM/batch-size ONLY).

NOT a real training run. TITAN is never invoked here -- we reuse the frozen 768-d
slide embeddings already cached by phase2_camelyon.py (cam16_titan_emb.pt: 100
CAMELYON16 slides, 60 normal / 40 tumor). Embeddings are fixed inputs; only a small
MLP projection head is trained, with a SupCon loss over label-based positives/
negatives. This sidesteps the ALiBi-recompute bottleneck found in the earlier
phase6 probe entirely, since no TITAN forward pass happens here at all.

Answers one question only: does this workload fit in VRAM, and at what batch size.
It says nothing about whether SupCon fine-tuning helps downstream metrics -- see
phase8/phase9/phase9b for that.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import random

import torch
import torch.nn.functional as F
from torch import nn

from common import RESULTS_DIR

EMB_CACHE = RESULTS_DIR / "cam16_titan_emb.pt"
OUT = RESULTS_DIR / "phase7_supcon_vram_smoke.json"

BATCH_SIZES = [8, 16, 32, 64, 128]
N_STEPS = 8
TEMPERATURE = 0.07
PROJ_DIM = 256
VRAM_BUDGET_GB = 20.0  # "comfortably under ~20GB on the 24GB card"


class ProjHead(nn.Module):
    def __init__(self, in_dim=768, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(z, labels, temperature=0.07):
    """Standard SupCon (Khosla et al. 2020), 1 view/sample: positives = other
    in-batch samples sharing the label, negatives = all other in-batch samples."""
    n = z.shape[0]
    sim = z @ z.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability
    not_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
    exp_sim = sim.exp() * not_self
    log_prob = sim - exp_sim.sum(dim=1, keepdim=True).log()

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same_label & not_self
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts
    return -mean_log_prob_pos.mean()


def balanced_batch(tumor_idx, normal_idx, batch_size, rng):
    """batch_size/2 tumor + batch_size/2 normal, no within-batch repeats.
    Returns None if the pool can't fill it (this is the class-imbalance check,
    independent of VRAM)."""
    half = batch_size // 2
    if half > len(tumor_idx) or (batch_size - half) > len(normal_idx):
        return None
    t = rng.sample(tumor_idx, half)
    n = rng.sample(normal_idx, batch_size - half)
    idx = t + n
    rng.shuffle(idx)
    return idx


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available -- this is a VRAM smoke test, refusing to run on CPU.")

    if not EMB_CACHE.exists():
        print(f"[phase7] {EMB_CACHE} not found -- run phase2_camelyon.py first.")
        return 1

    d = torch.load(EMB_CACHE)
    emb, labels, ids = d["emb"], d["labels"], d["ids"]
    print(f"[phase7] loaded cached embeddings: {tuple(emb.shape)}  "
          f"({int((labels==1).sum())} tumor / {int((labels==0).sum())} normal)")
    emb = emb.to(device)          # frozen inputs; requires_grad stays False
    labels = labels.to(device)

    tumor_idx = [i for i in range(len(ids)) if labels[i] == 1]
    normal_idx = [i for i in range(len(ids)) if labels[i] == 0]
    max_balanced_batch = 2 * min(len(tumor_idx), len(normal_idx))
    print(f"[phase7] tumor pool={len(tumor_idx)}  normal pool={len(normal_idx)}  "
          f"-> max balanced batch without replacement = {max_balanced_batch}")

    rng = random.Random(0)
    rows = []

    for bs in BATCH_SIZES:
        print(f"\n{'='*60}\n[phase7] batch_size={bs}\n{'='*60}", flush=True)
        row = {"batch_size": bs, "filled": None, "ok": None, "peak_alloc_gb": None,
               "peak_reserved_gb": None, "reason": None}

        first_batch = balanced_batch(tumor_idx, normal_idx, bs, rng)
        if first_batch is None:
            reason = (f"class imbalance: needs {bs//2} tumor slides, only "
                      f"{len(tumor_idx)} available (pool caps balanced batch at "
                      f"{max_balanced_batch})")
            print(f"  SKIPPED -- {reason}")
            row.update(filled=False, reason=reason)
            rows.append(row)
            continue
        row["filled"] = True

        head = ProjHead(768, PROJ_DIM).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok, err = True, None
        for step in range(N_STEPS):
            try:
                idx = balanced_batch(tumor_idx, normal_idx, bs, rng) if step > 0 else first_batch
                idx_t = torch.tensor(idx, device=device)
                x = emb[idx_t]                      # fixed frozen inputs
                y = labels[idx_t]
                z = head(x)                          # only the head has grad
                loss = supcon_loss(z, y, TEMPERATURE)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                print(f"  step {step+1}/{N_STEPS}: loss={loss.item():.4f}", flush=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    ok, err = False, str(e)
                    print(f"  step {step+1}: CUDA OOM", flush=True)
                    torch.cuda.empty_cache()
                    break
                raise

        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        row.update(ok=ok, peak_alloc_gb=peak_alloc, peak_reserved_gb=peak_reserved, reason=err)
        status = "OK" if ok else f"OOM ({err[:80]}...)"
        print(f"[phase7] batch={bs}: {status}  peak_alloc={peak_alloc:.4f}GB  "
              f"peak_reserved={peak_reserved:.4f}GB", flush=True)
        rows.append(row)

        del head, opt
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("[phase7] SUMMARY")
    print(f"  {'batch':>6s}  {'filled':>7s}  {'ok':>5s}  {'peak_alloc_GB':>14s}  {'peak_reserved_GB':>17s}")
    for r in rows:
        filled = "yes" if r["filled"] else "no"
        ok = "-" if r["ok"] is None else ("yes" if r["ok"] else "NO/OOM")
        pa = f"{r['peak_alloc_gb']:.4f}" if r["peak_alloc_gb"] is not None else "-"
        pr = f"{r['peak_reserved_gb']:.4f}" if r["peak_reserved_gb"] is not None else "-"
        print(f"  {r['batch_size']:>6d}  {filled:>7s}  {ok:>5s}  {pa:>14s}  {pr:>17s}")

    fitting = [r for r in rows if r["filled"] and r["ok"] and r["peak_reserved_gb"] < VRAM_BUDGET_GB]
    largest_ok = max((r["batch_size"] for r in fitting), default=None)
    print(f"\n  Largest batch size fitting comfortably under {VRAM_BUDGET_GB}GB: {largest_ok}")
    print(f"  Class-imbalance cap on balanced batch size (independent of VRAM): "
          f"{max_balanced_batch} (tumor pool={len(tumor_idx)})")

    with open(OUT, "w") as f:
        json.dump({"rows": rows, "max_balanced_batch": max_balanced_batch,
                    "largest_batch_under_budget": largest_ok,
                    "vram_budget_gb": VRAM_BUDGET_GB}, f, indent=2, default=str)
    print(f"\n[saved] {OUT}")
    print("[phase7] OK")


if __name__ == "__main__":
    main()
