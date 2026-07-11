"""Cache CONCHv1.5 patch features per BRACS ROI, so the slide encoder can be LoRA-fine-tuned
without re-running CONCH every step.

extract_bracs_features.py caches only the final 768-d TITAN embedding per ROI. Encoder LoRA
needs the *inputs* to the slide encoder: the CONCH patch features [N,768], their coords [N,2],
and patch_size_lv0. This script writes exactly those, reusing the SAME tiling, downsampling and
2x2 grid-padding as the frozen-embedding extractor, so a bf16 LoRA-off forward over this cache
reproduces the frozen baseline (verified in step 2).

Output: data/BRACS/features/patch/<item_id>.npz  with
    features : float32 [N, 768]   -- CONCH patch features, INCLUDING injected zero-pad patches
    coords   : int64   [N, 2]     -- (x, y) in 20x px, spacing = TILE
    n_real   : int                -- real tissue patches (features[:n_real] before padding)
plus data/BRACS/features/patch/meta.json versioning TILE, DOWNSAMPLE and a hash of the
pad/tile logic, so a stale cache can be detected.

Invariant asserted per ROI: the number of all-zero feature rows (patches TITAN's background
mask will drop) equals the number of zero-pad patches injected by _pad_grid_to_2x2 -- i.e. the
only zero patches are the grid pads, nothing real was silently zeroed.

Resumable: skips ROIs whose .npz already exists. Safe to Ctrl-C and re-run.
Env: same knobs as extract_bracs_features.py (DOWNSAMPLE, TILE, CONCH_BATCH, BLANK_*).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import hashlib
import inspect
import json
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import load_titan, get_device, TITAN_ROOT
import extract_bracs_features as ex
from extract_bracs_features import (roi_to_patches, _pad_grid_to_2x2, roi_path,
                                    TILE, DOWNSAMPLE, CONCH_BATCH)
import pandas as pd

Image.MAX_IMAGE_PIXELS = None

BRACS_ROOT = TITAN_ROOT / "data" / "BRACS"
MANIFEST = BRACS_ROOT / "manifest_roi.csv"
PATCH_DIR = BRACS_ROOT / "features" / "patch"
META_JSON = PATCH_DIR / "meta.json"


def _logic_hash():
    """Hash the source of the tiling/padding functions so a code change invalidates the cache."""
    src = "".join(inspect.getsource(f) for f in (roi_to_patches, _pad_grid_to_2x2))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


@torch.inference_mode()
def cache_roi(pil, conch, tfm, device):
    """ROI PIL -> (features[N,768] float32, coords[N,2] int64, n_real). Mirrors embed_roi up to
    the slide encoder; stops before encode_slide_from_patch_features."""
    tiles, coords = roi_to_patches(pil)
    x = torch.stack([tfm(t) for t in tiles])
    feats = []
    for i in range(0, len(x), CONCH_BATCH):
        with torch.autocast(device.type, torch.float16):
            feats.append(conch(x[i:i + CONCH_BATCH].to(device)).float().cpu())
    features = torch.cat(feats)                                   # (n_real, 768)
    n_real = len(tiles)
    features, coords = _pad_grid_to_2x2(features, coords)         # may append zero-feature rows
    coords_t = torch.tensor(coords, dtype=torch.long)

    # invariant: the only all-zero feature rows are the injected grid pads
    n_zero = int((features.abs().sum(dim=1) == 0).sum().item())
    n_pad = features.shape[0] - n_real
    assert n_zero == n_pad, (f"zero-row count {n_zero} != injected pads {n_pad} "
                             f"(a real patch was zeroed, or a pad is non-zero)")
    return features.numpy().astype(np.float32), coords_t.numpy().astype(np.int64), n_real


def main():
    sys.stdout.reconfigure(line_buffering=True)
    if not MANIFEST.exists():
        sys.exit(f"[patch-cache] {MANIFEST} not found -- run download_bracs.py --set roi first.")
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    missing = [r.item_id for r in manifest.itertuples() if not roi_path(r).exists()]
    if missing:
        sys.exit(f"[patch-cache] {len(missing)} ROI file(s) missing on disk (e.g. {missing[:3]}). "
                 f"Re-run download_bracs.py.")

    meta = {"tile": TILE, "downsample": DOWNSAMPLE, "logic_hash": _logic_hash(),
            "conch": "conch_v1_5", "n_rois": int(len(manifest))}
    if META_JSON.exists():
        old = json.loads(META_JSON.read_text())
        stale = {k: (old.get(k), meta[k]) for k in ("tile", "downsample", "logic_hash")
                 if old.get(k) != meta[k]}
        if stale:
            sys.exit(f"[patch-cache] existing meta.json disagrees with current code: {stale}. "
                     f"Delete {PATCH_DIR} to rebuild, or revert the code change.")

    print(f"[patch-cache] {len(manifest)} ROIs; TILE={TILE} DOWNSAMPLE={DOWNSAMPLE} "
          f"logic={meta['logic_hash']}")
    device = get_device()
    if device.type != "cuda":
        print("[patch-cache] WARNING: CUDA unavailable; CONCH on CPU is slow but will work.")
    model, device = load_titan(device)
    conch, eval_transform = model.return_conch()
    conch = conch.to(device).eval()

    todo = [r for r in manifest.itertuples() if not (PATCH_DIR / f"{r.item_id}.npz").exists()]
    print(f"[patch-cache] {len(manifest) - len(todo)} already cached, {len(todo)} to extract")

    counts = []
    for r in tqdm(todo, desc="caching", unit="roi"):
        pil = Image.open(roi_path(r)).convert("RGB")
        features, coords, n_real = cache_roi(pil, conch, eval_transform, device)
        np.savez(PATCH_DIR / f"{r.item_id}.npz", features=features, coords=coords,
                 n_real=np.int64(n_real))
        counts.append(features.shape[0])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if counts:
        c = np.array(counts)
        print(f"[patch-cache] patches/ROI (incl. pads) this run: min={c.min()} "
              f"median={int(np.median(c))} max={c.max()} mean={c.mean():.1f}")

    META_JSON.write_text(json.dumps(meta, indent=2))
    n_files = len(list(PATCH_DIR.glob("*.npz")))
    print(f"[patch-cache] {n_files} .npz files in {PATCH_DIR}")
    assert n_files == len(manifest), f"expected {len(manifest)} npz, found {n_files}"
    print(f"[patch-cache] meta.json written -> {META_JSON}")
    print("[patch-cache] OK")


if __name__ == "__main__":
    sys.exit(main())
