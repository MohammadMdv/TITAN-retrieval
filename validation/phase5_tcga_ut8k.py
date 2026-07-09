"""Phase 5: TCGA-UT-8K (32-class ROI benchmark) — SUBSET validation of TITAN.

TCGA-UT-8K is 25,495 ROIs of 8192x8192 px (377 GB raw, no precomputed features). We do a
bounded, class-balanced SUBSET: stream a few ROIs per class, extract a TITAN embedding for
each on the fly (tile -> CONCHv1.5 -> TITAN slide encoder), store only the 768-d embeddings
(raw images never persisted), then linear-probe. Result is subset-approximate vs paper 0.832.

Config via env:
  PER_CLASS_TRAIN (default 20), PER_CLASS_VAL (8), PER_CLASS_TEST (15)
  MAX_STREAM (hard cap on examples streamed per split, default 6000) -- runaway guard
  TILE (patch size in ROI px, default 512), CONCH_BATCH (default 128)
"""
import os
import sys
from collections import defaultdict, Counter

import numpy as np
import torch

from common import load_titan, load_hf_token, RESULTS_DIR, save_results, get_device

DATASET = "MahmoodLab/TCGA-UniformTumor-8K"
LABEL_FIELD = "cancer"
PATIENT_FIELD = "PATIENT"
EMB_CACHE = RESULTS_DIR / "tcga_ut8k_subset_emb.pt"


def tile_roi_to_embedding(pil_img, conch, eval_transform, model, device, tile=512, batch=128,
                          skip_white=True):
    """8192x8192 ROI -> one TITAN embedding via CONCHv1.5 patch features + slide encoder."""
    W, H = pil_img.size
    tiles, coords = [], []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            patch = pil_img.crop((x, y, x + tile, y + tile))
            if skip_white:
                arr = np.asarray(patch.convert("L"))
                if arr.mean() > 220 and arr.std() < 15:   # near-blank background tile
                    continue
            tiles.append(eval_transform(patch))
            coords.append((x, y))
    if len(tiles) == 0:  # fully blank fallback: keep the center tile
        patch = pil_img.crop((0, 0, tile, tile))
        tiles.append(eval_transform(patch)); coords.append((0, 0))

    x = torch.stack(tiles)
    feats = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            with torch.autocast(device.type, torch.float16):
                feats.append(conch(x[i:i + batch].to(device)).float().cpu())
    features = torch.cat(feats)                              # (n_patch, 768)
    coords_t = torch.tensor(coords, dtype=torch.long)        # level-0 px positions
    with torch.inference_mode(), torch.autocast(device.type, torch.float16):
        emb = model.encode_slide_from_patch_features(
            features.to(device), coords_t.to(device), tile).float().cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return emb.squeeze(0)                                     # (768,)


def build_subset(model, conch, eval_transform, device):
    if EMB_CACHE.exists():
        d = torch.load(EMB_CACHE)
        print(f"[phase5] loaded cached embeddings: { {k: len(v['y']) for k, v in d.items()} }")
        return d

    from datasets import load_dataset
    from tqdm import tqdm
    load_hf_token()

    quotas = {"train": int(os.environ.get("PER_CLASS_TRAIN", 20)),
              "val":   int(os.environ.get("PER_CLASS_VAL", 8)),
              "test":  int(os.environ.get("PER_CLASS_TEST", 15))}
    max_stream = int(os.environ.get("MAX_STREAM", 6000))
    tile = int(os.environ.get("TILE", 512))
    cbatch = int(os.environ.get("CONCH_BATCH", 128))

    out = {}
    for split, per_class in quotas.items():
        ds = load_dataset(DATASET, split=split, streaming=True)
        seen_class = Counter()
        embs, ys, pts = [], [], []
        n_stream = 0
        pbar = tqdm(ds, desc=f"{split} (<= {per_class}/class)")
        for ex in pbar:
            n_stream += 1
            if n_stream > max_stream:
                print(f"[phase5] {split}: hit MAX_STREAM={max_stream}"); break
            lab = ex[LABEL_FIELD]
            if seen_class[lab] >= per_class:
                continue
            emb = tile_roi_to_embedding(ex["image"].convert("RGB"), conch, eval_transform,
                                        model, device, tile=tile, batch=cbatch)
            embs.append(emb); ys.append(lab); pts.append(ex[PATIENT_FIELD])
            seen_class[lab] += 1
            pbar.set_postfix(collected=len(ys), classes=len(seen_class))
            # early stop: every class has its quota (needs a known class count; approx via plateau)
        out[split] = {"X": torch.stack(embs).numpy(), "y": np.array(ys), "patient": np.array(pts)}
        print(f"[phase5] {split}: {len(ys)} ROIs over {len(set(ys))} classes "
              f"(streamed {n_stream})")
        torch.save(out, EMB_CACHE)  # incremental
    return out


def run_linear_probe(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import log_loss
    from titan.utils import get_eval_metrics

    le = LabelEncoder().fit(np.concatenate([data[s]["y"] for s in data]))
    def enc(s): return data[s]["X"], le.transform(data[s]["y"]), data[s]["patient"]
    Xtr, ytr, ptr = enc("train"); Xva, yva, _ = enc("val"); Xte, yte, pte = enc("test")

    # patient-disjoint guardrail
    assert set(ptr).isdisjoint(pte), "patient leak train<->test"
    assert set(ptr).isdisjoint(data["val"]["patient"]), "patient leak train<->val"

    from joblib import Parallel, delayed
    grid = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    def fit(c):
        import warnings; warnings.filterwarnings("ignore")
        m = LogisticRegression(C=1.0 / c, max_iter=500, random_state=0, solver="lbfgs")
        m.fit(Xtr, ytr)
        return log_loss(yva, m.predict_proba(Xva), labels=np.arange(len(le.classes_))), m
    fits = Parallel(n_jobs=int(os.environ.get("N_JOBS", 10)))(delayed(fit)(c) for c in grid)
    model = fits[int(np.argmin([v for v, _ in fits]))][1]

    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)
    res = get_eval_metrics(yte, preds, probs, roc_kwargs={"multi_class": "ovo", "average": "macro"})
    return {k.strip("/"): float(v) for k, v in res.items() if isinstance(v, (int, float))}, len(le.classes_)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    device = get_device()
    model, device = load_titan(device)
    conch, eval_transform = model.return_conch()
    conch = conch.to(device).eval()

    data = build_subset(model, conch, eval_transform, device)
    print("[phase5] === LINEAR PROBE (subset; paper 32-class bal.acc = 0.832) ===")
    res, n_classes = run_linear_probe(data)
    print(f"[phase5] subset LP: bacc={res.get('bacc'):.4f} over {n_classes} classes "
          f"(train={len(data['train']['y'])}/val={len(data['val']['y'])}/test={len(data['test']['y'])})")

    save_results("phase5_tcga_ut8k.json", {
        "n_classes_in_subset": n_classes,
        "counts": {s: int(len(data[s]["y"])) for s in data},
        "linear_probe": res,
        "reference": {"full_32class_bacc": 0.832},
        "note": "SUBSET of 25,495 ROIs; approximate vs paper. Tiled 512px, patient-disjoint enforced.",
    })
    print("[phase5] OK")


if __name__ == "__main__":
    sys.exit(main())
