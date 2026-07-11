"""Extract one frozen TITAN slide embedding per BRACS ROI (CONCHv1.5 patches -> slide encoder).

Input : the ROI .png files downloaded by download_bracs.py, plus data/BRACS/manifest_roi.csv
        (item_id, patient_id, label, official_split).
Output: data/BRACS/features/bracs_roi_titan.pkl -- a DataFrame with columns
        [item_id, patient_id, label, split, n_patches, embedding(768,)], the drop-in input for
        a downstream retrieval / LoRA study. Nothing is trained here; TITAN stays frozen.

MAGNIFICATION -- the one thing that must be right
-------------------------------------------------
BRACS ROIs are stored at 40x / 0.25 um/px (Aperio AT2; per the BRACS paper). CONCHv1.5 inside
TITAN was trained on 512x512 patches at 20x / 0.5 um/px -- i.e. each patch covers 256 um of
tissue. Feeding a 40x ROI to CONCH at face value would present tissue at DOUBLE its trained
magnification and quietly corrupt every feature. So each ROI is DOWNSAMPLED 2x (40x -> 20x)
before tiling. After that a 512 px tile again covers 256 um, which is what CONCH expects, and
TITAN's patch_size_lv0 is 512 (its documented "512 @20x" convention).

This is the same failure mode as feeding 40x CAMELYON16 features to a 20x model -- see the
caveat on that experiment in RESULTS.md. Here it is corrected, not merely noted.

Tiling
------
BRACS ROIs are small and ragged (median ~1335x1489 px at 40x, range 327-12103), unlike the
uniform 8192 px TCGA-UT-8K ROIs. After the 2x downsample most ROIs yield only a handful of
512 px tiles, and some are smaller than a single tile. Partial edge tiles and sub-tile ROIs
are white-padded up to 512x512 (padding preserves um/px; upscaling would not). Near-blank
tiles are dropped; an all-blank ROI keeps one centered tile so it still yields an embedding.

Resumable: each ROI's embedding is cached to features/emb/<item_id>.npy and skipped if present.
Safe to Ctrl-C and re-run. The final pack step assembles the .pkl from those caches.

Env: DOWNSAMPLE (default 2), TILE (512), CONCH_BATCH (256), BLANK_MEAN (220), BLANK_STD (15).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from common import load_titan, get_device, TITAN_ROOT

Image.MAX_IMAGE_PIXELS = None  # BRACS ROIs are trusted; disable the decompression-bomb guard

BRACS_ROOT = TITAN_ROOT / "data" / "BRACS"
ROI_DIR = BRACS_ROOT / "roi"
MANIFEST = BRACS_ROOT / "manifest_roi.csv"
FEAT_DIR = BRACS_ROOT / "features"
EMB_CACHE = FEAT_DIR / "emb"                      # per-ROI .npy, for resumability
OUT_PKL = FEAT_DIR / "bracs_roi_titan.pkl"

DOWNSAMPLE = int(os.environ.get("DOWNSAMPLE", 2))   # 40x -> 20x
TILE = int(os.environ.get("TILE", 512))             # px at 20x; CONCH's native patch size
CONCH_BATCH = int(os.environ.get("CONCH_BATCH", 256))
BLANK_MEAN = float(os.environ.get("BLANK_MEAN", 220))
BLANK_STD = float(os.environ.get("BLANK_STD", 15))


def _pad_to(patch, size, fill=(255, 255, 255)):
    if patch.size == (size, size):
        return patch
    bg = Image.new("RGB", (size, size), fill)
    bg.paste(patch, (0, 0))
    return bg


def roi_to_patches(pil, tile=TILE, downsample=DOWNSAMPLE):
    """ROI (40x) -> list of (PIL 512x512 tile at 20x, (x,y) in downsampled px). None-skip blanks."""
    if downsample != 1:
        pil = pil.resize((max(1, pil.width // downsample), max(1, pil.height // downsample)),
                         Image.BILINEAR)
    W, H = pil.size
    tiles, coords = [], []
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            patch = pil.crop((x, y, min(x + tile, W), min(y + tile, H)))  # never reads past edge
            arr = np.asarray(patch.convert("L"))
            if arr.mean() > BLANK_MEAN and arr.std() < BLANK_STD:         # background -> drop
                continue
            tiles.append(_pad_to(patch, tile))
            coords.append((x, y))
    if not tiles:  # fully blank ROI: keep one tile so it still produces an embedding
        tiles.append(_pad_to(pil.crop((0, 0, min(tile, W), min(tile, H))), tile))
        coords.append((0, 0))
    return tiles, coords


def _pad_grid_to_2x2(features, coords):
    """Guarantee the patch grid spans at least 2x2 cells, so TITAN's ALiBi path never sees a
    degenerate grid.

    ~1/3 of BRACS ROIs are smaller than 2x2 patches after the 2x downsample (many are a single
    512 px tile). TITAN's get_alibi() indexes a meshgrid of shape (grid_H, grid_W) and raises
    IndexError when either is 1. The fix injects ZERO-feature patches at the missing corners of
    a 2x2 grid: preprocess_features() builds its background mask as `any(feature != 0)`, so a
    genuinely zero patch is marked background and dropped before attention -- it fixes the grid
    geometry without contributing to the embedding. Real tissue patches are untouched.
    """
    coords = [tuple(c) for c in coords]
    xs = {x for x, _ in coords}
    ys = {y for _, y in coords}
    if len(xs) >= 2 and len(ys) >= 2:
        return features, coords
    x0, y0 = min(xs), min(ys)
    for corner in [(x0 + TILE, y0), (x0, y0 + TILE), (x0 + TILE, y0 + TILE)]:
        if corner not in coords:
            coords.append(corner)
            features = torch.cat([features, torch.zeros(1, features.shape[1],
                                                        dtype=features.dtype)], dim=0)
    return features, coords


@torch.inference_mode()
def embed_roi(pil, conch, tfm, model, device):
    tiles, coords = roi_to_patches(pil)
    x = torch.stack([tfm(t) for t in tiles])
    feats = []
    for i in range(0, len(x), CONCH_BATCH):
        with torch.autocast(device.type, torch.float16):
            feats.append(conch(x[i:i + CONCH_BATCH].to(device)).float().cpu())
    features = torch.cat(feats)                                   # (n_patch, 768)
    n_real = len(tiles)
    features, coords = _pad_grid_to_2x2(features, coords)         # zero pads dropped by bg_mask
    coords_t = torch.tensor(coords, dtype=torch.long)            # 20x px, spacing = TILE
    with torch.autocast(device.type, torch.float16):
        emb = model.encode_slide_from_patch_features(
            features.to(device), coords_t.to(device), TILE).float().cpu()  # patch_size_lv0=TILE
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return emb.squeeze(0).numpy().astype(np.float32), n_real


def roi_path(row):
    """download_bracs.py lays files out as roi/<split>/<label>/<item_id>.png."""
    return ROI_DIR / row.official_split / row.label / f"{row.item_id}.png"


def main():
    sys.stdout.reconfigure(line_buffering=True)
    if not MANIFEST.exists():
        sys.exit(f"[feat] {MANIFEST} not found -- run download_bracs.py --set roi first.")
    EMB_CACHE.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    missing = [r.item_id for r in manifest.itertuples() if not roi_path(r).exists()]
    if missing:
        sys.exit(f"[feat] {len(missing)} ROI file(s) from the manifest are missing on disk "
                 f"(e.g. {missing[:3]}). Re-run download_bracs.py to finish the download.")
    print(f"[feat] {len(manifest)} ROIs across "
          f"{manifest.patient_id.nunique()} patients; splits={dict(manifest.official_split.value_counts())}")

    device = get_device()
    if device.type != "cuda":
        print("[feat] WARNING: CUDA unavailable; CONCH on CPU is slow but will work.")
    model, device = load_titan(device)
    conch, eval_transform = model.return_conch()
    conch = conch.to(device).eval()

    todo = [r for r in manifest.itertuples() if not (EMB_CACHE / f"{r.item_id}.npy").exists()]
    print(f"[feat] {len(manifest) - len(todo)} already cached, {len(todo)} to extract")

    patch_counts = []
    for r in tqdm(todo, desc="extracting", unit="roi"):
        pil = Image.open(roi_path(r)).convert("RGB")
        emb, n_patch = embed_roi(pil, conch, eval_transform, model, device)
        np.save(EMB_CACHE / f"{r.item_id}.npy", emb)
        patch_counts.append(n_patch)
    if patch_counts:
        pc = np.array(patch_counts)
        print(f"[feat] patches/ROI over this run: min={pc.min()} median={int(np.median(pc))} "
              f"max={pc.max()} mean={pc.mean():.1f}")

    # ---- pack every cached embedding into one DataFrame pickle
    print("[feat] packing embeddings ...")
    rows = []
    for r in manifest.itertuples():
        emb = np.load(EMB_CACHE / f"{r.item_id}.npy")
        rows.append({"item_id": r.item_id, "patient_id": r.patient_id, "label": r.label,
                     "split": r.official_split, "embedding": emb})
    df = pd.DataFrame(rows)
    assert df.embedding.iloc[0].shape == (768,), df.embedding.iloc[0].shape

    # patient-disjointness is a property of the split; re-assert it on the packed features so a
    # corrupted manifest can never silently produce a leaking feature set.
    g = {s: set(x.patient_id) for s, x in df.groupby("split")}
    for a in g:
        for b in g:
            if a < b:
                assert g[a].isdisjoint(g[b]), f"patient leak between {a} and {b}"

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT_PKL)
    print(f"[feat] wrote {len(df)} embeddings -> {OUT_PKL}")
    print(f"[feat] per split: {dict(df['split'].value_counts())}")
    print(f"[feat] per class: {dict(df['label'].value_counts())}")
    print("[feat] OK")


if __name__ == "__main__":
    sys.exit(main())
