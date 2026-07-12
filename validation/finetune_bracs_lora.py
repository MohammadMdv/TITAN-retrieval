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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common import load_titan, get_device, save_results, TITAN_ROOT
from retrieval_bracs import load_split, evaluate, chance_references, FEAT_PKL
from retrieval_common import cosine_sim, l2_normalize, eval_ranking


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """Cosine LR schedule with linear warmup. Copied from titan/finetune.py (which can't be
    imported here -- it pulls in omegaconf). Honors an optional per-group `lr_scale`."""
    def _assign(new_lr):
        for g in optimizer.param_groups:
            g["lr"] = new_lr * g.get("lr_scale", 1.0)

    def _adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e, es = step - warmup_length, steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / max(es, 1))) * base_lr
        _assign(lr)
        return lr

    return _adjuster

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


def mean_conch_df(manifest):
    """Mean-pool the cached CONCH patch features per ROI (masking zero-pads via n_real) ->
    DataFrame with the frozen-pkl schema. Throws away TITAN's aggregation entirely."""
    ids, embs = list(manifest["item_id"]), []
    for item_id in ids:
        features, _, n_real = load_patch(item_id)
        embs.append(features[:n_real].mean(axis=0).astype(np.float32))  # pads excluded
    return pd.DataFrame({
        "item_id": ids, "patient_id": list(manifest["patient_id"]),
        "label": list(manifest["label"]), "split": list(manifest["official_split"]),
        "embedding": embs,
    })


def confusion_matrix(sim, db_labels, q_labels, classes):
    """Top-1 retrieval confusion: rows = query true class, cols = retrieved top-1 class,
    values = fraction of that class's queries. Also returns raw counts."""
    top1 = db_labels[np.argmax(sim, axis=1)]
    idx = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(q_labels, top1):
        counts[idx[t], idx[p]] += 1
    frac = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    return counts, frac


def print_confusion(frac, classes, title):
    print(f"\n[lora] {title}  (row=true, col=top-1 retrieved; diagonal=Acc@1)")
    print("        " + "  ".join(f"{c:>5}" for c in classes))
    for i, c in enumerate(classes):
        row = "  ".join(f"{frac[i, j]:5.2f}" for j in range(len(classes)))
        print(f"  {c:>5} {row}")


def mode_confusion(args):
    """Step 0.1 + 0.3a (training-free): frozen-TITAN test confusion matrix, and mean-CONCH-kNN
    vs frozen-TITAN retrieval (the sharp 'does TITAN's aggregator beat naive mean-pool?' test).
    Runs on CPU from the pkl + patch cache; no training, no model load."""
    if not FEAT_PKL.exists():
        sys.exit(f"[lora] {FEAT_PKL} not found -- run extract_bracs_features.py first.")
    frozen = pd.read_pickle(FEAT_PKL)
    Xtr, ytr, ctr, _ = load_split(frozen, "train")
    Xte, yte, cte, _ = load_split(frozen, "test")
    classes = sorted(set(ytr))

    sim = cosine_sim(Xte, Xtr)
    counts, frac = confusion_matrix(sim, ytr, yte, classes)
    print_confusion(frac, classes, "FROZEN TITAN — test top-1 confusion")
    # atypical/borderline block the plan flagged: do ADH/UDH/DCIS mostly retrieve each other?
    block = ["ADH", "UDH", "DCIS"]
    bi = [classes.index(c) for c in block]
    for c, i in zip(block, bi):
        within = counts[i, bi].sum()
        total = counts[i].sum()
        print(f"[lora]   {c}: {within}/{total} = {within/max(total,1):.2f} of top-1 land in "
              f"{{ADH,UDH,DCIS}}")

    # mean-CONCH-kNN vs frozen-TITAN: if naive mean-pool ~= TITAN, the aggregator adds nothing.
    manifest = pd.read_csv(MANIFEST)
    mc = mean_conch_df(manifest)
    print("\n[lora] mean-CONCH-kNN (no TITAN aggregation, no training):")
    mc_results = run_eval(mc, "mean-conch-knn")
    print("[lora] frozen-TITAN test Acc@1=0.5053 (Step 2). If mean-CONCH ~= this, the "
          "aggregator is not the lever.")

    frozen_results = run_eval(frozen, "frozen-titan")
    save_results("finetune_bracs_lora_step0_confusion.json", {
        "classes": classes,
        "frozen_titan_test_confusion_counts": counts.tolist(),
        "frozen_titan_test_confusion_frac": frac.tolist(),
        "mean_conch_knn": mc_results,
        "frozen_titan": frozen_results,
    })
    print("[lora] Step-0 confusion + mean-CONCH-kNN saved.")
    print("[lora] OK")


# --------------------------------------------------------------------------------------------
# Proxy-Anchor metric-learning core (shared by the Step-0 controls and the Step-3 LoRA run)
# --------------------------------------------------------------------------------------------
class ProxyAnchor(nn.Module):
    """Proxy-Anchor loss (Kim et al., CVPR 2020). One learnable proxy per class.

    Proxies are L2-normalized every forward (so `emb @ P.T` is cosine). fp32 only: the
    exp(alpha*(cos +/- delta)) terms peak near exp(35) ~ 2e15, fine for fp32, overflow for fp16.
    Defaults alpha=32, delta=0.1 are the paper's. Proxies want a much higher LR than the map
    params (handled by a param group with lr_scale)."""
    def __init__(self, n_classes, dim, alpha=32.0, delta=0.1):
        super().__init__()
        self.proxies = nn.Parameter(torch.empty(n_classes, dim))
        nn.init.kaiming_normal_(self.proxies, a=np.sqrt(5))
        self.alpha, self.delta, self.C = alpha, delta, n_classes

    def forward(self, emb, y_idx):
        P = F.normalize(self.proxies, dim=1)
        cos = emb @ P.t()                                        # [B, C], emb already normalized
        onehot = F.one_hot(y_idx, self.C).float()               # [B, C]
        pos, neg = onehot.bool(), ~onehot.bool()
        pos_exp = torch.where(pos, torch.exp(-self.alpha * (cos - self.delta)), torch.zeros_like(cos))
        neg_exp = torch.where(neg, torch.exp(self.alpha * (cos + self.delta)), torch.zeros_like(cos))
        with_pos = onehot.sum(0) > 0                             # classes present in the batch
        n_pos = with_pos.sum().clamp(min=1)
        pos_term = (torch.log1p(pos_exp.sum(0))[with_pos]).sum() / n_pos
        neg_term = (torch.log1p(neg_exp.sum(0))).sum() / self.C
        return pos_term + neg_term


def pk_batches(y_idx, n_classes, K, rng, K_replace=True):
    """One epoch of PK batches with P = all classes present (so both loss terms stay non-empty).
    Each batch = K samples per class -> batch size n_classes*K. Yields index arrays."""
    by_class = [np.where(y_idx == c)[0] for c in range(n_classes)]
    steps = max(1, len(y_idx) // (n_classes * K))
    for _ in range(steps):
        idx = [rng.choice(ci, size=K, replace=K_replace or len(ci) < K) for ci in by_class]
        yield np.concatenate(idx)


def val_signal(lin, Xtr_t, ytr, ctr, Xva_t, yva, cva):
    """Smoothed val-retrieval selection score = mean(Acc@1, Acc@3), val queries vs train DB."""
    with torch.no_grad():
        Zdb = F.normalize(lin(Xtr_t), dim=1).cpu().numpy()
        Zva = F.normalize(lin(Xva_t), dim=1).cpu().numpy()
    m = eval_ranking(cosine_sim(Zva, Zdb), ytr, yva, k=3)
    return 0.5 * (m["acc@1"] + m["acc@3"]), m


def train_linear_map(Xtr, ytr, ctr, Xva, yva, cva, classes, device, seed,
                     epochs=40, lr=1e-3, proxy_mult=100.0, K=8, patience=6, wd=1e-2):
    """Train a single Linear(768->768) with Proxy-Anchor, selecting on val retrieval. The map is
    initialized to identity so before training the retrieval == the untrained baseline."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    D, C = Xtr.shape[1], len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    ytr_idx = np.array([cls_idx[c] for c in ytr])

    lin = nn.Linear(D, D).to(device)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(D)); lin.bias.zero_()        # identity init == no-op start
    loss_fn = ProxyAnchor(C, D).to(device)
    opt = torch.optim.AdamW(
        [{"params": lin.parameters(), "weight_decay": wd},
         {"params": loss_fn.parameters(), "weight_decay": 0.0, "lr_scale": proxy_mult}], lr=lr)
    steps_per_epoch = max(1, len(Xtr) // (C * K))
    sched = cosine_lr(opt, lr, warmup_length=int(steps_per_epoch * epochs * 0.1),
                      steps=steps_per_epoch * epochs)

    Xtr_t = torch.tensor(Xtr, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    ytr_idx_t = torch.tensor(ytr_idx, device=device)

    best_sig, best_state, bad, step = -1.0, None, 0, 0
    for epoch in range(epochs):
        lin.train()
        for idx in pk_batches(ytr_idx, C, K, rng):
            sched(step); step += 1
            emb = F.normalize(lin(Xtr_t[idx]), dim=1)
            loss = loss_fn(emb, ytr_idx_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        lin.eval()
        sig, m = val_signal(lin, Xtr_t, ytr, ctr, Xva_t, yva, cva)
        if sig > best_sig + 1e-4:
            best_sig, best_state, bad = sig, {k: v.clone() for k, v in lin.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    lin.load_state_dict(best_state)
    lin.eval()
    return lin, best_sig


def map_embed(lin, X, device):
    with torch.no_grad():
        return F.normalize(lin(torch.tensor(X, device=device)), dim=1).cpu().numpy()


def source_splits(source):
    """Return (Xtr,ytr,ctr, Xva,yva,cva, Xte,yte,cte, classes) for a control source."""
    if source == "frozen":
        df = pd.read_pickle(FEAT_PKL)
    elif source == "mean-conch":
        df = mean_conch_df(pd.read_csv(MANIFEST))
    else:
        raise ValueError(source)
    Xtr, ytr, ctr, _ = load_split(df, "train")
    Xva, yva, cva, _ = load_split(df, "val")
    Xte, yte, cte, _ = load_split(df, "test")
    return Xtr, ytr, ctr, Xva, yva, cva, Xte, yte, cte, sorted(set(ytr))


def mode_control(args):
    """Step 0.2/0.3b: trained Linear(768->768) + Proxy-Anchor on frozen embeddings and on
    mean-CONCH. Gives the ceiling of a metric-learning head in each space -- the bar LoRA must
    clear to justify touching the encoder. Multi-seed; reports test mean +/- std vs baseline."""
    device = get_device()
    sources = ["frozen", "mean-conch"] if args.source == "both" else [args.source]
    out = {"seeds": args.seeds, "baseline_frozen_test_acc@1": 0.5053}
    for source in sources:
        Xtr, ytr, ctr, Xva, yva, cva, Xte, yte, cte, classes = source_splits(source)
        print(f"\n[lora] === trained control: {source}  (db={len(ytr)} val={len(yva)} "
              f"test={len(yte)}, {args.seeds} seeds) ===")
        per_seed = []
        for seed in range(args.seeds):
            lin, best_sig = train_linear_map(Xtr, ytr, ctr, Xva, yva, cva, classes, device, seed)
            Zdb, Zte = map_embed(lin, Xtr, device), map_embed(lin, Xte, device)
            metrics, per_class = evaluate(Zdb, ytr, ctr, Zte, yte, cte, f"{source}-test")
            per_seed.append({"metrics": metrics, "per_class_acc@1": per_class,
                             "val_sig": best_sig})
            print(f"  seed {seed}: val_sig={best_sig:.4f}  test "
                  + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        keys = list(per_seed[0]["metrics"])
        mean = {k: float(np.mean([s["metrics"][k] for s in per_seed])) for k in keys}
        std = {k: float(np.std([s["metrics"][k] for s in per_seed])) for k in keys}
        print(f"  [{source}] test mean+/-std: "
              + "  ".join(f"{k}={mean[k]:.4f}+/-{std[k]:.4f}" for k in keys))
        out[source] = {"per_seed": per_seed, "test_mean": mean, "test_std": std}
    save_results("finetune_bracs_lora_step0_controls.json", out)
    print("\n[lora] Step-0 trained controls saved. Bar for LoRA = best of these vs 0.5053.")
    print("[lora] OK")


# --------------------------------------------------------------------------------------------
# Step 3 — Block-LoRA on the TITAN slide encoder + Proxy-Anchor
# --------------------------------------------------------------------------------------------
def block_targets(model):
    """The 24 block Linear names LoRA adapts: {attn.qkv, attn.proj, mlp.fc1, mlp.fc2} x 6."""
    import torch.nn as nn
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)
            and "blocks.modules_list." in n
            and (n.endswith(".attn.qkv") or n.endswith(".attn.proj")
                 or n.endswith(".mlp.fc1") or n.endswith(".mlp.fc2"))]


def reset_lora(peft_model):
    """Re-init the adapter to PEFT's default state: lora_A ~ kaiming_uniform, lora_B = 0.

    Called at the START of every seed, right after that seed's torch.manual_seed, so seed k's
    adapter is a pure function of k -- independent of the ambient process RNG and of whatever
    earlier seeds did. (PEFT does this init inside get_peft_model, which runs before any seeding,
    so relying on it makes the run unreproducible.)
    """
    for n, p in peft_model.named_parameters():
        if "lora_B" in n:
            nn.init.zeros_(p)
        elif "lora_A" in n:
            nn.init.kaiming_uniform_(p, a=np.sqrt(5))


def preload_cache(item_ids, device):
    """All patch features/coords on GPU (fp32, ~85 MB) so per-ROI forwards have no H2D cost."""
    cache = {}
    for iid in item_ids:
        f, c, _ = load_patch(iid)
        cache[iid] = (torch.tensor(f, dtype=torch.float32, device=device),
                      torch.tensor(c, dtype=torch.long, device=device))
    return cache


def enc_one(model, cache, iid, tile):
    """One ROI -> 768-d slide embedding (with grad if model is training)."""
    f, c = cache[iid]
    return model.encode_slide_from_patch_features(f, c, tile).squeeze(0)


@torch.no_grad()
def embed_ids(model, cache, ids, tile):
    model.eval()
    return np.stack([enc_one(model, cache, i, tile).float().cpu().numpy() for i in ids])


def retrieval_metrics(Zdb, ydb, cdb, Zq, yq, cq, name):
    m, per_class = evaluate(Zdb, ydb, cdb, Zq, yq, cq, name)
    return m, per_class


def top1_correct(Zdb, ydb, Zq, yq):
    sim = cosine_sim(Zq, Zdb)
    return ydb[np.argmax(sim, axis=1)] == yq


def paired_bootstrap(base_ok, lora_ok, n_boot=5000, seed=0):
    """Paired bootstrap CI of the Acc@1 difference (lora - baseline) over the same queries,
    plus McNemar discordant counts."""
    rng = np.random.default_rng(seed)
    diff = lora_ok.astype(int) - base_ok.astype(int)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    b = int(((base_ok) & (~lora_ok)).sum())    # baseline right, lora wrong
    c = int(((~base_ok) & (lora_ok)).sum())    # baseline wrong, lora right
    return {"delta_acc@1": float(diff.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "mcnemar_b_base_only": b, "mcnemar_c_lora_only": c}


def train_lora(model, peft_model, cache, tr_ids, ytr, ctr, va_ids, yva, cva, classes, tile,
               device, seed, args):
    """Train Block-LoRA + Proxy-Anchor, selecting on val retrieval; keep best-epoch adapter."""
    from copy import deepcopy
    rng = np.random.default_rng(seed)
    C = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    ytr_idx = np.array([cls_idx[c] for c in ytr])
    tr_ids = np.asarray(tr_ids)

    loss_fn = ProxyAnchor(C, 768).to(device)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "weight_decay": args.wd},
         {"params": loss_fn.parameters(), "weight_decay": 0.0, "lr_scale": args.proxy_mult}],
        lr=args.lr)
    steps_per_epoch = max(1, len(tr_ids) // (C * args.K))
    sched = cosine_lr(opt, args.lr, warmup_length=int(steps_per_epoch * args.epochs * 0.1),
                      steps=steps_per_epoch * args.epochs)

    best_sig, best_state, bad, step = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        for local in pk_batches(ytr_idx, C, args.K, rng):
            sched(step); step += 1
            batch_ids = tr_ids[local]
            embs = torch.stack([enc_one(model, cache, i, tile) for i in batch_ids])
            embs = F.normalize(embs, dim=1)
            y = torch.tensor(ytr_idx[local], device=device)
            loss = loss_fn(embs, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # val retrieval selection
        Zdb = embed_ids(model, cache, tr_ids, tile)
        Zva = embed_ids(model, cache, va_ids, tile)
        m = eval_ranking(cosine_sim(Zva, Zdb), ytr, yva, k=3)
        sig = 0.5 * (m["acc@1"] + m["acc@3"])
        print(f"    seed {seed} epoch {epoch:2d}: loss={loss.item():.3f} "
              f"val Acc@1={m['acc@1']:.4f} Acc@3={m['acc@3']:.4f} sig={sig:.4f}"
              + ("  *" if sig > best_sig + 1e-4 else ""))
        if sig > best_sig + 1e-4:
            best_sig = sig
            best_state = deepcopy({k: v.detach().cpu() for k, v in
                                   peft_model.state_dict().items() if "lora_" in k})
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            print(f"    seed {seed}: early stop at epoch {epoch}")
            break
    # restore best adapter
    peft_model.load_state_dict(best_state, strict=False)
    model.eval()
    return best_sig


def mode_lora(args):
    from peft import LoraConfig, get_peft_model
    meta = load_meta(); tile = meta["tile"]
    manifest = pd.read_csv(MANIFEST)
    def split_ids(s):
        d = manifest[manifest.official_split == s]
        return list(d.item_id), d.label.values, d.patient_id.values
    tr_ids, ytr, ctr = split_ids("train")
    va_ids, yva, cva = split_ids("val")
    te_ids, yte, cte = split_ids("test")
    classes = sorted(set(ytr))
    print(f"[lora] Step 3 Block-LoRA: db={len(tr_ids)} val={len(va_ids)} test={len(te_ids)} "
          f"r={args.r} alpha={args.alpha} lr={args.lr} epochs={args.epochs} seeds={args.seeds}")

    device = get_device()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, device = load_titan(device)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = block_targets(model)
    cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=targets, bias="none")
    peft_model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] LoRA on {len(targets)} Linears; trainable params={n_train:,}")

    print("[lora] preloading patch cache to GPU ...")
    cache = preload_cache(tr_ids + va_ids + te_ids, device)

    # fp32 LoRA-off baseline (paired-test reference; adapter disabled == frozen encoder)
    with peft_model.disable_adapter():
        Zdb0 = embed_ids(model, cache, tr_ids, tile)
        Zte0 = embed_ids(model, cache, te_ids, tile)
    base_m, base_pc = retrieval_metrics(Zdb0, ytr, ctr, Zte0, yte, cte, "lora-off")
    base_ok = top1_correct(Zdb0, ytr, Zte0, yte)
    print(f"[lora] fp32 LoRA-off test: " + " ".join(f"{k}={v:.4f}" for k, v in base_m.items()))

    seeds_out = []
    for seed in range(args.seeds):
        print(f"\n[lora] --- seed {seed} ---")
        torch.manual_seed(seed)          # the ONE seeding point for this seed: adapter init,
        np.random.seed(seed)             # proxy init, LoRA dropout and PK sampling all follow
        reset_lora(peft_model)           # fresh adapter, drawn from the just-seeded generator
        best_sig = train_lora(model, peft_model, cache, tr_ids, ytr, ctr, va_ids, yva, cva,
                              classes, tile, device, seed, args)
        Zdb = embed_ids(model, cache, tr_ids, tile)
        Zte = embed_ids(model, cache, te_ids, tile)
        m, pc = retrieval_metrics(Zdb, ytr, ctr, Zte, yte, cte, f"lora-s{seed}")
        lora_ok = top1_correct(Zdb, ytr, Zte, yte)
        paired = paired_bootstrap(base_ok, lora_ok)
        print(f"[lora] seed {seed} test: " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
              + f"  | dAcc@1={paired['delta_acc@1']:+.4f} CI{paired['ci95']}")
        seeds_out.append({"seed": seed, "val_sig": best_sig, "metrics": m,
                          "per_class_acc@1": pc, "paired_vs_loraoff": paired})

    keys = list(seeds_out[0]["metrics"])
    mean = {k: float(np.mean([s["metrics"][k] for s in seeds_out])) for k in keys}
    std = {k: float(np.std([s["metrics"][k] for s in seeds_out])) for k in keys}
    print("\n[lora] ===== SUMMARY =====")
    print(f"[lora] fp32 LoRA-off baseline: " + " ".join(f"{k}={v:.4f}" for k, v in base_m.items()))
    print(f"[lora] LoRA test mean+/-std:   "
          + "  ".join(f"{k}={mean[k]:.4f}+/-{std[k]:.4f}" for k in keys))
    save_results("finetune_bracs_lora_step3.json", {
        "config": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout, "lr": args.lr,
                   "epochs": args.epochs, "seeds": args.seeds, "K": args.K, "scope": "block"},
        "baseline_loraoff_fp32": {"metrics": base_m, "per_class_acc@1": base_pc},
        "public_baseline_frozen_fp16_acc@1": 0.5053,
        "per_seed": seeds_out, "test_mean": mean, "test_std": std})
    print("[lora] Step-3 results saved.")
    print("[lora] OK")


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
    ap.add_argument("--mode", default="baseline",
                    choices=["baseline", "confusion", "control", "lora"])
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--source", default="both", choices=["frozen", "mean-conch", "both"],
                    help="control mode: which feature space to train the linear map on")
    ap.add_argument("--seeds", default=3, type=int, help="control/lora: number of seeds")
    # lora hyperparameters
    ap.add_argument("--r", default=8, type=int)
    ap.add_argument("--alpha", default=16, type=int)
    ap.add_argument("--dropout", default=0.05, type=float)
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--wd", default=1e-2, type=float)
    ap.add_argument("--proxy_mult", default=100.0, type=float)
    ap.add_argument("--K", default=8, type=int, help="samples per class per PK batch")
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--patience", default=5, type=int)
    args = ap.parse_args()
    if args.mode == "baseline":
        mode_baseline(args)
    elif args.mode == "confusion":
        mode_confusion(args)
    elif args.mode == "control":
        mode_control(args)
    elif args.mode == "lora":
        mode_lora(args)


if __name__ == "__main__":
    sys.exit(main())
