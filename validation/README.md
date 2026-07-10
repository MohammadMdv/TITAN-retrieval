# TITAN validation

Reproduces the published TITAN zero-shot / linear-probe / retrieval numbers, then investigates
whether **patient-disjoint slide retrieval** can be improved beyond the frozen model.

- `PLAN.md` — original scoping/design notes.
- `RESULTS.md` — numbers, compared against paper reference values.
- `results/` — JSON metric summaries per experiment (checkpoints/embeddings are gitignored).

## Setup

```
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env   # fill in HF_TOKEN (needs the gated MahmoodLab/TITAN license accepted)
python validation/download_assets.py   # pulls TITAN weights + TCGA_TITAN_features.pkl
```

Optional env vars (see `common.py`): `CAMELYON16_ROOT` (needed only by the scripts that use
local CAMELYON16 CONCHv1.5 features), and `VALIDATION_RESULTS_DIR` to write outputs elsewhere.

## Scripts

Named `<category>_<dataset>[_<technique>].py`.

| Script | What it does |
|---|---|
| `smoke_slide_encoding.py` | Slide-encoding smoke test on one local CAMELYON16 h5 |
| `classification_camelyon16.py` | CAMELYON16 zero-shot + linear probe (binary tumor/normal) |
| `classification_tcga_ot.py` | TCGA-OT (46-class) zero-shot + linear probe |
| `classification_tcga_ut8k.py` | TCGA-UT-8K (32-class ROI) linear probe, subset |
| `retrieval_tcga_ot.py` | **Baseline**: TCGA-OT retrieval (patient-disjoint), raw cosine |
| `retrieval_tcga_ot_whitening.py` | Tier-1: centering / PCA / PCA-whitening / ZCA |
| `retrieval_tcga_ot_kreciprocal.py` | Tier-1: k-reciprocal re-ranking (Zhong et al. 2017) |
| `retrieval_tcga_ot_query_expansion.py` | Tier-1: αQE and DBA (reported separately) |
| `retrieval_camelyon16.py` | TITAN retrieval vs. an external PathSearch comparison run |
| `subtasks_tcga_ot.py` | Harder patient-disjoint TCGA-OT sub-typing tasks |
| `common.py`, `retrieval_common.py` | Shared paths/model loading; shared retrieval protocol + metrics |
| `download_assets.py`, `ut8k_label_recon.py` | Asset download; TCGA-UT-8K label recon |
| `download_bracs.py` | Patient-disjoint, budget-capped BRACS WSI subset download (FTP, resumable) |
| `make_results.py` | Aggregates `results/*.json` into `RESULTS.md` |

`run_all.sh` runs the classification, baseline-retrieval, and Tier-1 experiments.

## Retrieval protocol

Every retrieval experiment uses the identical protocol, so deltas isolate the technique:

- **database** = train split, minus any slide whose `case_id` appears in val or test
- **val queries** = val split — used *only* to select hyperparameters
- **test queries** = test split — used *once*, to report the selected config

Patient (`case_id`) disjointness between the database and both query sets is asserted at load
time. The database is fixed, so a config tuned on val transfers to test unchanged.

## Findings

**Tier-1 (training-free post-processing) does not improve retrieval.** Whitening, k-reciprocal
re-ranking, and αQE/DBA were each tested in isolation on the frozen embeddings. None beat the
raw-cosine baseline on Acc@3. Notably:

- k-reciprocal's val-optimal λ is **1.0**, which by construction *is* plain cosine — the sweep
  chose "don't re-rank." (λ=1.0 reproducing the baseline exactly is also the implementation's
  sanity check, and it passes.)
- Full whitening (`pcaw-*`, `zca`) actively *hurts* Acc@3, implying TITAN's high-variance
  directions carry discriminative signal rather than nuisance variation — the usual motivation
  for whitening does not hold in this vision-language-aligned space.
- All three showed a small MVAcc@3 gain **on val** (+0.010–0.013). Re-selecting on MVAcc@3 and
  re-testing showed those gains **do not transfer to test** — they were val-set noise.

Taken with the earlier (now removed) SupCon projection-head experiment, which also failed, the
picture is consistent: **TITAN's frozen embedding geometry is already near-optimal for this
task**, and neither a learned head on top of it nor a training-free re-ranking of its output
improves patient-disjoint retrieval on TCGA-OT. Improving further likely requires adapting the
encoder itself (LoRA/adapters) or domain-adaptive continued pretraining on a large target-domain
corpus — not post-hoc manipulation of its output vectors.
