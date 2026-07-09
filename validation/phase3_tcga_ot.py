"""Phase 3: TCGA-OT (46-class, full WSIs) zero-shot + linear probe.

Runs entirely off the precomputed TCGA_TITAN_features.pkl (TITAN slide embeddings
for the whole TCGA cohort) + the ./datasets splits. No WSI/patch work.

Targets: linear-probe balanced accuracy ~= 0.704.
"""
import os
# 1 BLAS thread/process so the 10 parallel C-fits don't oversubscribe the 10 cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import yaml

from common import (load_titan, load_hf_token, CONFIG_TCGA_OT, DATASETS,
                    get_device, save_results)


def get_pkl_path():
    from huggingface_hub import hf_hub_download
    load_hf_token()
    return hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl")


def load_embeddings():
    path = get_pkl_path()
    with open(path, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame({"slide_id": data["filenames"],
                       "embeddings": list(np.asarray(data["embeddings"]))})
    return df


def run_zeroshot(model, device, task_config, test_csv, label_dict, target):
    from titan.utils import TEMPLATES, get_eval_metrics, bootstrap
    from tqdm import tqdm

    class_prompts = task_config["prompts"]
    sorted_cp = dict(sorted(class_prompts.items(),
                            key=lambda kv: label_dict.get(kv[0], float("inf"))))
    classes = list(sorted_cp.keys())
    prompts = [sorted_cp[k] for k in classes]
    with torch.autocast(device.type, torch.float16), torch.inference_mode():
        classifier = model.zero_shot_classifier(prompts, TEMPLATES, device=device)

    df = load_embeddings()
    df = df[df["slide_id"].isin(test_csv["slide_id"])].reset_index(drop=True)
    probs, targets = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="zero-shot"):
        emb = torch.as_tensor(row["embeddings"]).to(device)
        with torch.autocast(device.type, torch.float16), torch.inference_mode():
            probs.append(model.zero_shot(emb, classifier).cpu())
        lab = test_csv.loc[test_csv["slide_id"] == row["slide_id"], target].values[0]
        targets.append(label_dict[lab])
    probs_all = torch.cat(probs, dim=0).float()
    targets_all = torch.tensor(targets)
    preds_all = probs_all.argmax(dim=1)
    res = get_eval_metrics(targets_all.numpy(), preds_all.numpy(), probs_all.numpy(),
                           roc_kwargs={"multi_class": "ovo", "average": "macro"})
    res = {k.strip("/"): float(v) for k, v in res.items() if isinstance(v, (int, float))}
    print(f"[phase3] ZS point estimate -> bacc={res.get('bacc'):.4f}  auroc={res.get('auroc')}")
    boot_n = int(os.environ.get("BOOTSTRAP_N", "1000"))
    boot = {}
    if boot_n > 0:
        mean, std = bootstrap(results_dict={"targets": targets_all.numpy(),
                                            "preds": preds_all.numpy(),
                                            "probs": probs_all.numpy()}, n=boot_n)
        boot = {k.split("/")[-1]: f"{mean[k]:.4f} ± {std[k]:.4f}" for k in mean}
    return res, boot


def _fit_one(log2_coeff, X_tr, y_tr, X_val, y_val, max_iter):
    """Fit LR for one C and return (val_log_loss, model). Mirrors titan.eval_linear_probe
    exactly: C = 1/log2_coeff, lbfgs, best-C chosen later by minimum val log_loss."""
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    lr = LogisticRegression(C=1.0 / log2_coeff, fit_intercept=True, max_iter=max_iter,
                            random_state=0, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    return log_loss(y_val, lr.predict_proba(X_val)), lr


def run_linear_probe(label_dict, target, max_iter=500):
    """Parallel version of titan.eval_linear_probe's sweep (same logic, 10x faster on CPU).
    Selection is identical: best C = min val log_loss; test metrics via get_eval_metrics."""
    from joblib import Parallel, delayed
    from titan.utils import get_eval_metrics, seed_torch
    import torch
    seed_torch(torch.device("cpu"), 0)  # match reference determinism

    emb_df = load_embeddings()
    def split(name):
        s = pd.read_csv(DATASETS / f"tcga-ot_{name}.csv")
        m = pd.merge(emb_df, s, on="slide_id")
        X = np.stack(m["embeddings"].values)
        y = m[target].apply(lambda x: label_dict[x]).values
        return X, y
    X_tr, y_tr = split("train")
    X_val, y_val = split("val")
    X_te, y_te = split("test")
    print(f"[phase3] LP sizes train={len(y_tr)} val={len(y_val)} test={len(y_te)}")

    log_spaced = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)  # paper grid
    n_jobs = int(os.environ.get("N_JOBS", "10"))
    print(f"[phase3] sweeping 45 C-values across {n_jobs} cores...")
    fits = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(c, X_tr, y_tr, X_val, y_val, max_iter) for c in log_spaced)
    best_i = int(np.argmin([vl for vl, _ in fits]))
    best_C = log_spaced[best_i]
    model = fits[best_i][1]
    print(f"[phase3] Best C (=1/{best_C:.3g}): 1/{best_C:.6g}")

    test_preds = model.predict(X_te)
    test_probs = model.predict_proba(X_te)  # 46 classes -> full matrix
    results = get_eval_metrics(y_te, test_preds, test_probs,
                               roc_kwargs={"multi_class": "ovo", "average": "macro"})
    results = {k.strip("/"): float(v) for k, v in results.items() if isinstance(v, (int, float))}
    outputs = {"targets": y_te, "preds": test_preds, "probs": test_probs}
    return results, outputs


def main():
    with open(CONFIG_TCGA_OT) as f:
        task_config = yaml.load(f, Loader=yaml.FullLoader)
    label_dict = task_config["label_dict"]
    target = task_config["target"]
    test_csv = pd.read_csv(DATASETS / "tcga-ot_test.csv")

    from titan.utils import bootstrap

    # --- Linear probe first: pure sklearn on precomputed pkl embeddings, no GPU/model. ---
    print("[phase3] === LINEAR PROBE (target bacc ~0.704) ===")
    lp, lp_outputs = run_linear_probe(label_dict, target)
    print(f"[phase3] LP point estimate -> bacc={lp.get('bacc'):.4f} (ref 0.704)  "
          f"auroc={lp.get('auroc')}  kappa={lp.get('kappa'):.4f}")
    out = {"linear_probe": lp, "reference": {"linear_probe_bacc": 0.704}}
    save_results("phase3_tcga_ot.json", out)  # save headline BEFORE the slow bootstrap

    # bootstrap CIs are a follow-on; N configurable (BOOTSTRAP_N=0 to skip). Doesn't gate LP.
    boot_n = int(os.environ.get("BOOTSTRAP_N", "1000"))
    if boot_n > 0:
        print(f"[phase3] running {boot_n}x bootstrap for LP CIs...")
        mean, std = bootstrap(results_dict=lp_outputs, n=boot_n)
        lp_boot = {k.split("/")[-1]: f"{mean[k]:.4f} ± {std[k]:.4f}" for k in mean}
        print(f"[phase3] LP bootstrap: {lp_boot}")
        out["linear_probe_bootstrap"] = lp_boot
        save_results("phase3_tcga_ot.json", out)

    # --- Zero-shot needs the model (GPU). Skip gracefully if CUDA is down. ---
    import torch
    if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
        print("[phase3] CUDA unavailable -> skipping zero-shot (needs the model). "
              "LP result saved. Recover GPU and re-run for zero-shot.")
        print("[phase3] OK (linear probe only)")
        return

    print("[phase3] === ZERO-SHOT ===")
    model, device = load_titan()
    zs, zs_boot = run_zeroshot(model, device, task_config, test_csv, label_dict, target)
    print(f"[phase3] ZS: bacc={zs.get('bacc'):.4f} auroc={zs.get('auroc')}")
    print(f"[phase3] ZS bootstrap: {zs_boot}")

    out.update({"zero_shot": zs, "zero_shot_bootstrap": zs_boot})
    save_results("phase3_tcga_ot.json", out)
    print("[phase3] OK")


if __name__ == "__main__":
    sys.exit(main())
