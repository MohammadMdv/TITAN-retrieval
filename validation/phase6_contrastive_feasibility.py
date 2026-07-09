"""Phase 6 (feasibility probe, NOT a real training run): can this GPU/RAM train a
small contrastive projection head on top of FROZEN TITAN slide embeddings?

This does not answer "is this a good idea for retrieval" -- it only answers
"does standard-config InfoNCE fine-tuning of a projection head fit in 24GB VRAM /
31GB system RAM, and how long does one step take, on the 100-slide CAMELYON16
subset we already have cached." Do not read the printed loss as evidence of
learned retrieval quality.

Design (TITAN frozen; only the head trains):
  - TITAN's slide encoder is a 12-layer/12-head ViT-style transformer with dense
    O(N^2) attention (verified in modeling_titan.py / vision_transformer.py). It
    runs under torch.no_grad() here -- no backward pass through it -- so its
    memory profile is identical to what Phase 2 already proved fits (same
    MAX_PATCHES cap idea). Only the small projection head needs gradients.
  - Two augmented "views" per slide = two independent random subsamples of that
    slide's cached CONCH patch features (already on disk in the h5 files from
    Phase 2 -- no CONCH re-encoding needed). This is the WSI-bag analogue of
    SimCLR's random-crop augmentation.
  - Projection head: Linear(768,768)-ReLU-Linear(768,proj_dim), L2-normalized --
    SimCLR's default head shape.
  - Loss: NT-Xent / InfoNCE, in-batch negatives, temperature=0.07 (CLIP/TITAN's
    own convention). Batches are class-stratified (normal+tumor both present) so
    same-class negatives are naturally present as "hard" negatives. Explicit kNN
    hard-negative mining would reuse the same cached embedding bank (~free, no
    extra encoder passes) so it would not change the memory/time profile tested
    here -- this probe already exercises the expensive part.
  - The slide encoder takes ONE slide at a time (variable N, dense attention), so
    views are encoded sequentially even within a "batch" -- batch size changes
    step wall-clock, not peak GPU memory. The real memory lever is max_patches
    per view (the O(N^2) term), same as Phase 2's MAX_PATCHES cap.

Two configs profiled back to back, a handful of steps each:
  STANDARD: batch=32 slides/step, max_patches=10000 (Phase 2's own cap), proj_dim=256
  MINIMUM:  batch=8  slides/step, max_patches=4000,  proj_dim=128

Env overrides: N_STEPS (default 3), STANDARD_ONLY=1, MINIMUM_ONLY=1.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import csv
import time
import random
import resource

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import h5py

from common import load_titan, get_device, save_results, CAM16_H5, CAM16_REF


def load_all_patch_features():
    """Preload every slide's CONCH patch features+coords to CPU RAM. Verified cheap:
    100 slides, 380-23413 patches each, ~1.8GB total as float32."""
    ref = {}
    with open(CAM16_REF) as f:
        for r in csv.DictReader(f):
            ref[r["image"].replace(".tif", "")] = r["type"].strip().lower()

    data = {}
    for h5p in sorted(CAM16_H5.glob("*.h5")):
        sid = h5p.stem
        if sid not in ref:
            continue
        with h5py.File(h5p, "r") as f:
            feat_key = "features" if "features" in f else "feats"
            feats = torch.from_numpy(f[feat_key][:])
            coords = torch.from_numpy(f["coords"][:])
        data[sid] = {"feat": feats, "coords": coords, "label": ref[sid]}
    return data


def sample_view(rec, max_patches, keep_frac, seed):
    n = rec["feat"].shape[0]
    k = min(max_patches, max(1, int(n * keep_frac)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].sort().values
    return rec["feat"][idx], rec["coords"][idx]


@torch.no_grad()
def encode_frozen(model, feat, coords, device):
    """Frozen TITAN slide-encoder forward. no_grad (not inference_mode) is
    deliberate: inference_mode tensors can't later feed a grad-enabled op (the
    head), so this uses the standard frozen-backbone pattern instead."""
    f_ = feat.to(device, non_blocking=True)
    c_ = coords.to(device, non_blocking=True)
    with torch.autocast(device.type, torch.float16):
        emb = model.encode_slide_from_patch_features(f_, c_, 512).float()
    del f_, c_
    return emb.squeeze(0)  # (768,) on device, requires_grad=False


class ProjHead(nn.Module):
    def __init__(self, in_dim=768, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def nt_xent(z, temperature=0.07):
    """SimCLR NT-Xent. z: (2B, D); views [0:B] and [B:2B] are the positive pairs."""
    n = z.shape[0]
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(n, device=z.device, dtype=torch.bool), -1e9)
    b = n // 2
    targets = torch.cat([torch.arange(b, n), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


def run_config(name, model, device, data, ids, batch_size, max_patches, keep_frac,
                proj_dim, n_steps):
    print(f"\n{'='*60}\n[phase6] CONFIG {name}: batch={batch_size} max_patches={max_patches} "
          f"keep_frac={keep_frac} proj_dim={proj_dim} steps={n_steps}\n{'='*60}", flush=True)

    head = ProjHead(768, proj_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    normals = [s for s in ids if data[s]["label"] == "normal"]
    tumors = [s for s in ids if data[s]["label"] == "tumor"]
    rng = random.Random(0)

    step_times, encode_times, head_times, gpu_peaks = [], [], [], []
    ok, err = True, None
    for step in range(n_steps):
        try:
            t0 = time.time()
            half = batch_size // 2
            batch_ids = rng.sample(normals, min(half, len(normals))) + \
                        rng.sample(tumors, min(batch_size - half, len(tumors)))
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            t_enc0 = time.time()
            emb_a, emb_b = [], []
            for i, sid in enumerate(batch_ids):
                fa, ca = sample_view(data[sid], max_patches, keep_frac, seed=step * 1000 + i)
                fb, cb = sample_view(data[sid], max_patches, keep_frac, seed=step * 1000 + i + 500)
                emb_a.append(encode_frozen(model, fa, ca, device))
                emb_b.append(encode_frozen(model, fb, cb, device))
            t_enc1 = time.time()

            z = torch.stack(emb_a + emb_b)   # frozen embeddings, requires_grad=False
            z = head(z)                       # only the head is in the autograd graph
            loss = nt_xent(z, temperature=0.07)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            t1 = time.time()

            gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            ram_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB

            step_times.append(t1 - t0); encode_times.append(t_enc1 - t_enc0)
            head_times.append(t1 - t_enc1); gpu_peaks.append(gpu_peak)
            print(f"  step {step+1}/{n_steps}: loss={loss.item():.4f} "
                  f"total={t1-t0:.2f}s (encode={t_enc1-t_enc0:.2f}s head={t1-t_enc1:.3f}s) "
                  f"gpu_peak={gpu_peak:.2f}GB ram={ram_peak:.2f}GB", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                ok, err = False, f"CUDA OOM: {e}"
                print(f"  step {step+1}: CUDA OOM", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            raise
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
            print(f"  step {step+1}: FAILED -- {err}", flush=True)
            break

    n_epoch_steps = -(-len(ids) // batch_size)  # ceil
    summary = {
        "ok": ok, "error": err,
        "n_steps_completed": len(step_times),
        "mean_step_s": float(np.mean(step_times)) if step_times else None,
        "mean_encode_s": float(np.mean(encode_times)) if encode_times else None,
        "mean_head_s": float(np.mean(head_times)) if head_times else None,
        "gpu_peak_gb": float(max(gpu_peaks)) if gpu_peaks else None,
        "steps_per_epoch_full_dataset": n_epoch_steps,
        "est_epoch_time_s": float(np.mean(step_times) * n_epoch_steps) if step_times else None,
    }
    if summary["est_epoch_time_s"]:
        print(f"[phase6] {name}: ~{summary['est_epoch_time_s']:.0f}s/epoch "
              f"({n_epoch_steps} steps over {len(ids)} slides)", flush=True)
    return summary


def main():
    device = get_device()
    model, device = load_titan(device)
    model.requires_grad_(False)

    print("[phase6] preloading cached CONCH patch features for all 100 slides ...", flush=True)
    data = load_all_patch_features()
    ids = sorted(data.keys())
    print(f"[phase6] loaded {len(ids)} slides "
          f"({sum(v['label']=='tumor' for v in data.values())} tumor / "
          f"{sum(v['label']=='normal' for v in data.values())} normal)", flush=True)

    n_steps = int(os.environ.get("N_STEPS", 3))
    results = {}

    if os.environ.get("MINIMUM_ONLY") != "1":
        results["standard"] = run_config("STANDARD", model, device, data, ids,
            batch_size=32, max_patches=10000, keep_frac=0.85, proj_dim=256, n_steps=n_steps)

    if os.environ.get("STANDARD_ONLY") != "1":
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        results["minimum"] = run_config("MINIMUM", model, device, data, ids,
            batch_size=8, max_patches=4000, keep_frac=0.7, proj_dim=128, n_steps=n_steps)

    print("\n" + "="*60)
    print("[phase6] FEASIBILITY SUMMARY")
    for name, r in results.items():
        status = "OK" if r["ok"] else f"FAILED ({r['error']})"
        print(f"  {name:10s}: {status}")
        if r["ok"]:
            print(f"             step={r['mean_step_s']:.2f}s (encode={r['mean_encode_s']:.2f}s, "
                  f"head={r['mean_head_s']:.3f}s)  gpu_peak={r['gpu_peak_gb']:.2f}GB  "
                  f"est_epoch={r['est_epoch_time_s']:.0f}s")

    save_results("phase6_contrastive_feasibility.json", results)
    print("[phase6] OK")


if __name__ == "__main__":
    sys.exit(main())
