"""Shared helpers for the SupCon projection-head experiment (Phase 8 fine-tune / Phase 9 eval).

Both scripts operate purely on the cached TCGA_TITAN_features.pkl embeddings already used
by phase3_tcga_ot.py / phase4_retrieval.py -- no TITAN forward pass, no WSI work, no new
download. TITAN itself is never touched; only a small head trains on its frozen outputs.

Note on naming: this is Supervised Contrastive (SupCon, Khosla et al. 2020) fine-tuning,
not self-supervised pretraining -- it uses the same OncoTreeCode labels the linear probe
targets. The question phase9 answers is "does a non-linear, label-aware projection of the
frozen embedding beat a plain linear probe on the raw embedding", not "does unsupervised
SSL pretraining help".
"""
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from common import load_hf_token, DATASETS

PROJ_DIM = 768


def load_embeddings():
    from huggingface_hub import hf_hub_download
    load_hf_token()
    path = hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return pd.DataFrame({"slide_id": data["filenames"],
                         "embeddings": list(np.asarray(data["embeddings"]))})


def build_split(emb_df, name, target="OncoTreeCode"):
    """Mirrors phase4_retrieval.build_split exactly: X (N,768) float32, raw string labels,
    case_id (for patient-disjoint checks), slide_id."""
    s = pd.read_csv(DATASETS / f"tcga-ot_{name}.csv")
    m = pd.merge(emb_df, s, on="slide_id")
    X = np.stack(m["embeddings"].values).astype(np.float32)
    labels = m[target].values
    cases = m["case_id"].values
    ids = m["slide_id"].values
    return X, labels, cases, ids


class ProjHead(nn.Module):
    """SimCLR/SupCon-style head: Linear-ReLU-Linear, L2-normalized output. The only
    trainable component in this whole experiment -- TITAN stays frozen throughout."""
    def __init__(self, in_dim=768, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(z, labels, temperature=0.07):
    """Standard SupCon (Khosla et al. 2020), 1 embedding/sample: positives = other in-batch
    samples sharing the label, negatives = all other in-batch samples. Anchors with zero
    in-batch positives contribute 0 (excluded from the effective signal, not a NaN/crash) --
    this is why phase8 uses a PK sampler rather than plain random batches, to keep that from
    happening often across the 46 OncoTreeCode classes."""
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
