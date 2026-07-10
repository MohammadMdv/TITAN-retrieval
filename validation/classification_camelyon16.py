"""CAMELYON16 classification: binary (tumor vs normal) with TITAN slide embeddings.

Uses the 100 local slides with CONCHv1.5 patch features (60 normal / 40 tumor,
labels from reference.csv incl. the CAMELYON16 test-set ground truth).

Steps:
  1. Encode every slide -> TITAN slide embedding (cached to scratchpad).
  2. Zero-shot: custom lymph-node prompts (indicative only; not a TITAN-designed task).
  3. Linear probe: stratified 60/40 split + bootstrap CI.
"""
import csv
import sys
import numpy as np
import torch

from common import (load_titan, encode_slide_from_h5, CAM16_H5, CAM16_REF,
                    get_device, save_results, RESULTS_DIR)

# label map: normal=0, tumor=1
LABELS = {"normal": 0, "tumor": 1}

# custom zero-shot prompts (CAMELYON16 = breast cancer mets in lymph nodes)
CLASS_PROMPTS = [
    ["normal lymph node", "benign lymph node tissue", "lymph node with no tumor",
     "lymph node without metastasis"],                                   # 0 normal
    ["lymph node metastasis", "metastatic carcinoma in lymph node",
     "lymph node with tumor", "metastatic breast carcinoma in lymph node"],  # 1 tumor
]


def load_labels():
    ref = {}
    with open(CAM16_REF) as f:
        for r in csv.DictReader(f):
            ref[r["image"].replace(".tif", "")] = r["type"].strip().lower()
    return ref


def build_embeddings(model, device):
    cache = RESULTS_DIR / "cam16_titan_emb.pt"
    part = RESULTS_DIR / "cam16_titan_emb.partial.pt"
    ref = load_labels()
    if cache.exists():
        d = torch.load(cache)
        print(f"[cls-cam16] loaded cached embeddings: {d['emb'].shape}")
        return d["emb"], d["labels"], d["ids"]

    # resumable: per-slide dict {sid: emb_vector}, flushed after every slide
    done = torch.load(part) if part.exists() else {}
    if done:
        print(f"[cls-cam16] resuming; {len(done)} slides already encoded")
    h5_files = sorted(CAM16_H5.glob("*.h5"))
    from tqdm import tqdm
    capped = []
    for h5 in tqdm(h5_files, desc="encoding CAM16 slides"):
        sid = h5.stem
        if sid not in ref or sid in done:
            continue
        emb, _, was_capped = encode_slide_from_h5(model, h5, device, patch_size_lv0=512)
        done[sid] = emb.squeeze(0)
        if was_capped:
            capped.append(sid)
        torch.save(done, part)

    ids = [h.stem for h in h5_files if h.stem in ref]
    emb = torch.stack([done[s] for s in ids])
    labels = torch.tensor([LABELS[ref[s]] for s in ids])
    torch.save({"emb": emb, "labels": labels, "ids": ids, "capped": capped}, cache)
    part.unlink(missing_ok=True)
    print(f"[cls-cam16] encoded {emb.shape[0]} slides -> {tuple(emb.shape)}; capped={len(capped)}")
    return emb, labels, ids


def run_zeroshot(model, device, emb, labels):
    from titan.utils import TEMPLATES, get_eval_metrics
    with torch.autocast(device.type, torch.float16), torch.inference_mode():
        classifier = model.zero_shot_classifier(CLASS_PROMPTS, TEMPLATES, device=device)
    probs = []
    with torch.autocast(device.type, torch.float16), torch.inference_mode():
        for e in emb:
            probs.append(model.zero_shot(e.to(device), classifier).cpu())
    probs = torch.cat(probs, dim=0).float()
    preds = probs.argmax(dim=1)
    res = get_eval_metrics(labels.numpy(), preds.numpy(), probs.numpy(), roc_kwargs={})
    return {k.strip("/"): float(v) for k, v in res.items() if isinstance(v, (int, float))}


def run_linear_probe(emb, labels, ids):
    from sklearn.model_selection import train_test_split
    from titan.eval_linear_probe import train_and_evaluate_logistic_regression_with_val
    from titan.utils import bootstrap

    X = emb.numpy()
    y = labels.numpy()
    # stratified: train / val / test = 50 / 20 / 30 (val used for C selection)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.6, stratify=y_tmp, random_state=0)
    print(f"[cls-cam16] LP split train={len(y_tr)} val={len(y_val)} test={len(y_te)}")
    log_spaced = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    results, outputs = train_and_evaluate_logistic_regression_with_val(
        X_tr, y_tr, X_val, y_val, X_te, y_te, log_spaced_values=log_spaced)
    results = {k.strip("/"): float(v) for k, v in results.items() if isinstance(v, (int, float))}
    # bootstrap CI on the test set
    mean, std = bootstrap(results_dict=outputs, n=1000, alpha=0.95)
    boot = {k.split("/")[-1]: f"{mean[k]:.4f} ± {std[k]:.4f}" for k in mean}
    return results, boot


def main():
    device = get_device()
    model, device = load_titan(device)
    emb, labels, ids = build_embeddings(model, device)
    n_pos = int(labels.sum())
    print(f"[cls-cam16] {len(labels)} slides: {len(labels)-n_pos} normal / {n_pos} tumor")

    zs = run_zeroshot(model, device, emb, labels)
    print(f"[cls-cam16] ZERO-SHOT: bacc={zs.get('bacc'):.3f} auroc={zs.get('auroc')}")

    lp, boot = run_linear_probe(emb, labels, ids)
    print(f"[cls-cam16] LINEAR PROBE: bacc={lp.get('bacc'):.3f} auroc={lp.get('auroc')}")
    print(f"[cls-cam16] LP bootstrap: {boot}")

    save_results("classification_camelyon16.json", {
        "n_slides": len(labels), "n_normal": len(labels) - n_pos, "n_tumor": n_pos,
        "zero_shot": zs, "linear_probe": lp, "linear_probe_bootstrap": boot,
        "note": "CAMELYON16 is not a TITAN-designed task; zero-shot indicative only.",
    })
    print("[cls-cam16] OK")


if __name__ == "__main__":
    sys.exit(main())
