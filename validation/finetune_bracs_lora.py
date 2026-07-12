"""LoRA fine-tuning of the TITAN slide encoder for BRACS ROI retrieval.

Built up step by step (see the plan). Currently implements:
  --mode baseline   Step 2: re-embed every ROI from the cached CONCH patch features through the
                    TRAINING numerical path (bf16, LoRA disabled), compare to the frozen fp16
                    embeddings (extract_bracs_features.py), and run the exact retrieval_bracs
                    evaluation. This is the real "0.505 to beat" and the proof that the patch
                    cache + slide-encoder forward reproduce the frozen baseline.

Reuses:
  - the patch cache from cache_bracs_patch_features.py (data/BRACS/features/patch/<id>.npz)
  - retrieval_bracs.{load_split, evaluate, per_class_acc1, chance_references}
  - common.{load_titan, save_results, TITAN_ROOT}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common import load_titan, get_device, save_results, TITAN_ROOT
from retrieval_bracs import load_split, evaluate, chance_references, FEAT_PKL

BRACS_ROOT = TITAN_ROOT / "data" / "BRACS"
MANIFEST = BRACS_ROOT / "manifest_roi.csv"
PATCH_DIR = BRACS_ROOT / "features" / "patch"
META_JSON = PATCH_DIR / "meta.json"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_meta():
    if not META_JSON.exists():
        sys.exit(f"[lora] {META_JSON} not found -- run cache_bracs_patch_features.py first.")
    return json.loads(META_JSON.read_text())


def load_patch(item_id):
    """Cached CONCH patch features for one ROI: (features[N,768] f32, coords[N,2] i64, n_real)."""
    z = np.load(PATCH_DIR / f"{item_id}.npz")
    return z["features"], z["coords"], int(z["n_real"])


@torch.inference_mode()
def embed_items(model, item_ids, tile, device, dtype=torch.bfloat16):
    """Re-embed a list of ROIs from the patch cache through the slide encoder.

    Zero-pad patches stay in `features`; TITAN's preprocess_features drops them via its
    `any(feature != 0)` background mask, exactly as during frozen extraction. Returns an
    [len(item_ids), 768] float32 array aligned to item_ids order.
    """
    out = np.empty((len(item_ids), 768), dtype=np.float32)
    autocast = torch.autocast(device.type, dtype) if dtype != torch.float32 \
        else torch.autocast(device.type, enabled=False)
    for i, item_id in enumerate(tqdm(item_ids, desc="embed", unit="roi", leave=False)):
        features, coords, _ = load_patch(item_id)
        f = torch.from_numpy(features).to(device)
        c = torch.from_numpy(coords).to(device)
        with autocast:
            emb = model.encode_slide_from_patch_features(f, c, tile)
        out[i] = emb.float().squeeze(0).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def embed_manifest(model, manifest, tile, device, dtype):
    """Embed every ROI in the manifest -> DataFrame with the same schema as the frozen pkl."""
    ids = list(manifest["item_id"])
    embs = embed_items(model, ids, tile, device, dtype)
    return pd.DataFrame({
        "item_id": ids,
        "patient_id": list(manifest["patient_id"]),
        "label": list(manifest["label"]),
        "split": list(manifest["official_split"]),
        "embedding": list(embs),
    })


def cosine_vs_frozen(df_new):
    """Per-item cosine between the re-embedded vectors and the frozen fp16 pkl embeddings."""
    if not FEAT_PKL.exists():
        return None
    old = pd.read_pickle(FEAT_PKL).set_index("item_id")["embedding"]
    cos = []
    for r in df_new.itertuples():
        a = np.asarray(r.embedding, dtype=np.float64)
        b = np.asarray(old.loc[r.item_id], dtype=np.float64)
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    cos = np.array(cos)
    return {"min": float(cos.min()), "mean": float(cos.mean()),
            "p01": float(np.percentile(cos, 1)), "n": int(len(cos))}


def run_eval(df, tag):
    """Run the exact retrieval_bracs protocol (train=db, val/test=queries) on a re-embedded df."""
    Xtr, ytr, ctr, _ = load_split(df, "train")
    Xva, yva, cva, _ = load_split(df, "val")
    Xte, yte, cte, _ = load_split(df, "test")
    print(f"[lora:{tag}] db(train)={len(ytr)} val_q={len(yva)} test_q={len(yte)} "
          f"classes={sorted(set(ytr))}")
    chance = chance_references(ytr, yte)
    results = {"n": {"db": len(ytr), "val": len(yva), "test": len(yte)},
               "classes": sorted(set(ytr)), "chance": chance}
    for split_name, (Xq, yq, cq) in [("val", (Xva, yva, cva)), ("test", (Xte, yte, cte))]:
        metrics, per_class = evaluate(Xtr, ytr, ctr, Xq, yq, cq, split_name)
        results[split_name] = {"metrics": metrics, "per_class_acc@1": per_class}
        print(f"[lora:{tag}] === {split_name.upper()} (n={len(yq)}) ===")
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print("  per-class Acc@1: " + "  ".join(f"{c}={v:.3f}" for c, v in per_class.items()))
    return results


def mode_baseline(args):
    meta = load_meta()
    tile = meta["tile"]
    dtype = DTYPES[args.dtype]
    manifest = pd.read_csv(MANIFEST)
    missing = [i for i in manifest["item_id"] if not (PATCH_DIR / f"{i}.npz").exists()]
    if missing:
        sys.exit(f"[lora] {len(missing)} patch caches missing (e.g. {missing[:3]}).")

    print(f"[lora] Step 2 baseline: {len(manifest)} ROIs, dtype={args.dtype}, "
          f"tile={tile}, LoRA disabled")
    device = get_device()
    model, device = load_titan(device)

    df = embed_manifest(model, manifest, tile, device, dtype)

    cos = cosine_vs_frozen(df)
    if cos is not None:
        print(f"[lora] cosine vs frozen fp16 pkl: mean={cos['mean']:.6f} "
              f"p01={cos['p01']:.6f} min={cos['min']:.6f} (n={cos['n']})")

    results = run_eval(df, f"baseline-{args.dtype}")
    results["dtype"] = args.dtype
    results["cosine_vs_frozen_fp16"] = cos
    results["meta"] = meta
    save_results(f"finetune_bracs_lora_baseline_{args.dtype}.json", results)
    print(f"[lora] Step-2 baseline saved. This bf16 LoRA-off number is what LoRA must beat.")
    print("[lora] OK")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="baseline", choices=["baseline"])
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    args = ap.parse_args()
    if args.mode == "baseline":
        mode_baseline(args)


if __name__ == "__main__":
    sys.exit(main())
