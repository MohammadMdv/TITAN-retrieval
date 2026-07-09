# TITAN validation

Reproduces the published TITAN zero-shot/linear-probe/retrieval numbers, then tests whether
a SupCon-trained projection head on top of frozen TITAN embeddings improves retrieval.

- `PLAN.md` — original scoping/design notes for the validation phases.
- `RESULTS.md` — numbers, compared against paper reference values.
- `results/` — small JSON metric summaries per phase (checkpoints/embeddings are gitignored).

## Setup

```
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env   # fill in HF_TOKEN (needs the gated MahmoodLab/TITAN license accepted)
python validation/download_assets.py   # pulls TITAN weights + TCGA_TITAN_features.pkl
```

Optional env vars (see `common.py`): `CAMELYON16_ROOT` (only needed for phases 1/2/6/7 and
`retrieval_camelyon_demo.py`, which use local CAMELYON16 CONCHv1.5 features), and
`VALIDATION_RESULTS_DIR` to write outputs somewhere other than `validation/results/`.

## Phases

| Script | What it does |
|---|---|
| `phase1_smoke.py` | Slide-encoding smoke test on one local CAMELYON16 h5 |
| `phase2_camelyon.py` | CAMELYON16 zero-shot + linear probe (binary tumor/normal) |
| `phase3_tcga_ot.py` | TCGA-OT (46-class) zero-shot + linear probe, off precomputed embeddings |
| `phase4_retrieval.py` | TCGA-OT slide retrieval (patient-disjoint), Acc@3 / MVAcc@3 |
| `phase5_tcga_ut8k.py` | TCGA-UT-8K (32-class ROI) linear probe, subset |
| `phase6_contrastive_feasibility.py` | Feasibility probe: does InfoNCE fine-tuning fit in VRAM (TITAN forward pass included) |
| `phase7_supcon_vram_smoke.py` | Same VRAM question, but for a head-only SupCon step (no TITAN forward pass) |
| `phase8_finetune_supcon.py` | Trains a projection head on frozen TCGA-OT embeddings with SupCon |
| `phase9_eval_before_after.py` | Before/after LP + retrieval eval of the phase 8 head vs. raw embeddings |
| `phase9b_checkpoint_sweep.py` | Same eval swept across phase 8's per-epoch checkpoints |
| `tcga_subtasks.py` | Harder patient-disjoint TCGA-OT sub-typing tasks (extra, off the same pkl) |
| `retrieval_camelyon_demo.py` | TITAN retrieval vs. an external PathSearch comparison run |
| `make_results.py` | Aggregates `results/*.json` into `RESULTS.md` |

`run_all.sh` runs phases 2→4 plus `make_results.py`.

## Finding: SupCon fine-tuning doesn't improve retrieval

Phases 8/9/9b test fine-tuning a small frozen-TITAN projection head with SupCon (a
*supervised* contrastive loss, using the same OncoTreeCode labels the linear probe targets —
not self-supervised pretraining). Result: the head overfits within a handful of epochs, and
retrieval Acc@3/MVAcc@3 degrade monotonically with further training while linear-probe
accuracy stays roughly flat. See `RESULTS.md` for the full sweep. Likely cause: SupCon
optimizes batch-local relative similarity (128 in-batch samples), while the retrieval metric
depends on global all-pairs structure across the whole database — the two aren't the same
objective, and a free nonlinear head has little to keep it aligned with the latter.
