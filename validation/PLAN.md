# TITAN Validation Plan (zero-shot + linear-probe + retrieval)

Goal: validate `mahmoodlab/TITAN` exactly as designed — zero-shot slide classification,
linear-probe slide classification, and slide retrieval — reproducing the published
reference numbers before the model is used for anything else.

## Reference numbers to reproduce
| Task | Metric | Target |
|---|---|---|
| TCGA-OT (46-class, full WSIs) | linear-probe balanced acc | **0.704** |
| TCGA-OT retrieval | Acc@3 / MVAcc@3 | **0.880 / 0.807** |
| TCGA-UT-8K (32-class ROI) | linear-probe balanced acc | **0.832** (subset ≈ approximate) |

## Environment facts (verified 2026-07-04)
- GPU: NVIDIA TITAN RTX, 24 GB, currently free. Disk: 264 GB free on `/`.
- `uv` at `~/.local/bin/uv`. System Python 3.12 — **too new** for TITAN's pins
  (`torch==2.0.1`, `transformers==4.46.0`). Use a **Python 3.10** venv via `uv`.
- HF token: read from `/home/user01/histopath-retrieval/.env` (`HF_TOKEN=...`).
  ⚠️ The token's HF account **must have accepted the gated license** at
  huggingface.co/MahmoodLab/TITAN or every model/feature download 401s. Verify first.
- TITAN repo: `/home/user01/TITAN`. Splits in `./datasets` are **already patient-disjoint**
  (verified: train∩val = train∩test = val∩test = 0 case_ids).

## Key design facts (drive the plan)
- TITAN = CONCHv1.5 patch encoder → TITAN transformer slide encoder → 1 global slide embedding.
- Slide encoding API: `model.encode_slide_from_patch_features(features, coords, patch_size_lv0)`.
  Inputs = CONCHv1.5 patch features + patch coords + `patch_size_level0` (1024 @40x, 512 @20x).
- Zero-shot: `model.zero_shot_classifier(prompts, TEMPLATES)` then `model.zero_shot(emb, clf)`.
- Linear probe: `titan.eval_linear_probe.train_and_evaluate_logistic_regression_with_val`.
- **TCGA-OT needs NO WSI/patch work** — precomputed `TCGA_TITAN_features.pkl` holds TITAN
  slide embeddings for the whole TCGA cohort (small). Zero-shot / LP / retrieval all run off it.

## Local data already present (no download)
- `pathsearch-project/PathSearch/data/CAMELYON16/h5_files/conch_v1_5/` — 100 slides,
  CONCHv1.5 **patch features + coords** (~18 MB each). Labels in `.../reference.csv`:
  **41 normal, 31 tumor** (labeled) + 28 `test_*` (**unlabeled** — exclude from eval).
  This is exactly the input `encode_slide_from_patch_features` needs → local smoke test
  AND a real WSI-level binary tumor/normal signal, with only the model to download.

---

## ⛔ Downloads to hand to the user (do NOT run these yourself)
Run inside the activated venv, with `HF_TOKEN` exported.
1. **TITAN + CONCHv1.5 weights** (gated): triggered by
   `AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)` — a few GB, cached to `~/.cache/huggingface`.
2. **`TCGA_TITAN_features.pkl`** via `hf_hub_download("MahmoodLab/TITAN","TCGA_TITAN_features.pkl")` — small (~tens of MB).
3. *(optional Phase 5)* **TCGA-UT-8K subset** — via the Phase-5 script below, NOT the full 377 GB.

---

## Phase 0 — Environment
1. `uv venv --python 3.10 /home/user01/TITAN/.venv` ; activate.
2. Install (hand the exact command to the user):
   `uv pip install -e /home/user01/TITAN` (uses TITAN's `setup.py` pins) plus
   `uv pip install huggingface_hub` if not pulled in.
   - If `torch==2.0.1` fails to resolve a CUDA wheel for this box, fall back to a
     CUDA-11.8/12.1 build of a nearby torch 2.x and keep `transformers==4.46.0` fixed
     (the remote modeling code targets that transformers version).
3. Smoke: `python -c "import torch; print(torch.cuda.is_available())"` → must be `True`.
**Exit criteria:** venv imports torch+CUDA, `import titan`, transformers 4.46.0.

## Phase 1 — Model load + local slide-encoding smoke test
1. Login with `HF_TOKEN`; `AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)`,
   `model.return_conch()` — confirms gated access works.
2. Load ONE local h5 (`.../CAMELYON16/h5_files/conch_v1_5/normal_001.h5`).
   **Verify** it has `features`, `coords`, and `coords.attrs['patch_size_level0']`.
   If the attr is missing, infer `patch_size_lv0` from CAMELYON16 magnification (40x→1024).
3. `encode_slide_from_patch_features(...)` under `torch.autocast('cuda', float16)` →
   one slide embedding of expected dim. No NaNs.
**Exit criteria:** full CONCHv1.5→TITAN slide-encoding path runs on real local features.

## Phase 2 — CAMELYON16 binary validation (real signal, local only)
Scope: 72 labeled slides (41 normal / 31 tumor); exclude the 28 unlabeled `test_*`.
1. Encode all 72 → slide-embedding matrix; cache to `scratchpad/cam16_titan_emb.pt`.
2. **Zero-shot**: custom prompts (e.g. "lymph node, no tumor" vs "lymph node metastasis,
   tumor present") through `zero_shot_classifier`/`zero_shot`; report balanced acc + AUROC.
   (Note: CAMELYON16 is NOT a TITAN-designed task/prompt — treat zero-shot as indicative only.)
3. **Linear probe**: stratified split of the 72 (e.g. 60/40, or 5-fold) →
   `train_and_evaluate_logistic_regression_with_val`; report bacc/AUROC + bootstrap CI.
   CAMELYON16 slides are patient-independent (no case grouping needed).
**Exit criteria:** pipeline produces sane binary numbers (tumor/normal well above chance).
This is the "initial test on the available dataset" and gates moving to TCGA.

## Phase 3 — TCGA-OT reproduction (primary, off precomputed pkl)
Load `TCGA_TITAN_features.pkl` → `{filenames, embeddings}`. Merge with `datasets/tcga-ot_{train,val,test}.csv`
on `slide_id`; labels = `OncoTreeCode` via `config_tcga-ot.yaml['label_dict']` (46 classes).
1. **Zero-shot** (mirror `zeroshot_demo.ipynb`): build classifier from `config['prompts']`
   + `TEMPLATES`; classify test set; `get_eval_metrics` (bacc, AUROC ovo, kappa) + bootstrap.
2. **Linear probe** (mirror `linear_probe_demo.ipynb`): train on train, select C on val,
   evaluate on test with `train_and_evaluate_logistic_regression_with_val`. Use the paper's
   full C-grid `np.logspace(np.log10(10e-6), np.log10(10e5), 45)` for the real run.
   **Target: balanced acc ≈ 0.704** (± bootstrap CI). Save results to `scratchpad/tcga_ot_lp.json`.
**Exit criteria:** LP bacc within CI of 0.704; zero-shot reported for reference.

## Phase 4 — TCGA-OT slide retrieval (patient-disjoint)
Not in the demos — implement from the same pkl embeddings.
1. **Database = train split; queries = test split.** Both already case-disjoint from each
   other AND from val (verified). Also drop any DB slide whose `case_id` appears in val/test.
   Assert `set(db.case_id) ∩ set(query.case_id) == ∅` before scoring.
2. L2-normalize embeddings; cosine similarity query×DB; exclude self; take top-K (K=3).
3. **Acc@3** = ≥1 of top-3 shares the query's `OncoTreeCode`.
   **MVAcc@3** = majority label of top-3 equals query label.
   **Target: 0.880 / 0.807.** Save to `scratchpad/tcga_ot_retrieval.json`.
**Exit criteria:** Acc@3/MVAcc@3 close to 0.880/0.807 with the disjointness assertion passing.

## Phase 5 — TCGA-UT-8K ROI benchmark (OPTIONAL, only after user go-ahead)
TCGA-UT-8K = **ROI dataset**: 25,495 regions of 8192×8192 px, **377 GB, raw images, no
precomputed features** → full download is impractical here. Do a representative subset.
1. **Subset-download script** (hand run to user): use `datasets.load_dataset(..., streaming=True)`
   to pull only a bounded subset — e.g. N ROIs per class across all 32 classes, capped to a
   storage budget (target ≪ 50 GB) — writing images to `scratchpad/tcga_ut8k_subset/`.
   Log exactly which ROIs/classes were taken (results are subset-approximate, not the full 0.832).
2. **Feature extraction** (GPU): for each 8192×8192 ROI, tile into CONCHv1.5 patches
   (per repo/TRIDENT convention), run CONCHv1.5, then `encode_slide_from_patch_features`
   to get one TITAN embedding per ROI. Cache embeddings; **delete raw ROI images after** to
   reclaim storage. Respect patient/slide-disjoint splits within the subset.
3. **Linear probe** over the subset embeddings; compare trend to 0.832 (note subset caveat).
**Exit criteria:** LP on the subset runs end-to-end; report with explicit "subset of N/25495" caveat.

---

## Storage & disjointness guardrails (apply throughout)
- Precomputed pkl + CAMELYON16 features are cheap; only Phase 5 raw ROIs are large →
  stream a subset, extract features, delete raws.
- Every retrieval/eval must assert **patient (`case_id`) disjointness** between the query/eval
  set and the retrieval database before computing any metric.
- Cache all intermediate embeddings under the session scratchpad, not in the repo.

## Deliverables
- Reusable scripts under `TITAN/validation/` (env, phase2_camelyon, phase3_tcga_ot,
  phase4_retrieval, phase5_tcga_ut8k_subset).
- `scratchpad/*.json` results per phase + a short `RESULTS.md` comparing to reference numbers.
