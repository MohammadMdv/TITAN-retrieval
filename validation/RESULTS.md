# TITAN Validation Results

Reference numbers from the TITAN Nature Medicine paper / repo README.

## Classification — CAMELYON16 binary (tumor vs normal)
- 100 slides (60 normal / 40 tumor). _Caveat: 512px@40x features (CONCHv1.5 expects ~20x); not a TITAN-designed task._

| Setting | Balanced acc | AUROC |
|---|---|---|
| Zero-shot | 0.5083 | -1.0000 |
| Linear probe | 0.9028 | 0.9630 |
- LP bootstrap: {'acc': '0.9003 ± 0.0548', 'bacc': '0.9027 ± 0.0557', 'kappa': '0.7900 ± 0.1146', 'nw_kappa': '0.7900 ± 0.1146', 'weighted_f1': '0.9009 ± 0.0543', 'loss': '0.6070 ± 0.4296', 'auroc': '0.9633 ± 0.0394'}

## Classification — TCGA-OT (46-class, full WSIs)
| Setting | Balanced acc | Reference | AUROC | Kappa |
|---|---|---|---|---|
| Linear probe | 0.7026 | **0.704** | 0.9896 | 0.8080 |
| Zero-shot | 0.6150 | — | 0.9823 | 0.6984 |
- LP bootstrap: {'acc': '0.7803 ± 0.0110', 'bacc': '0.7043 ± 0.0195', 'kappa': '0.8097 ± 0.0179', 'nw_kappa': '0.8188 ± 0.0140', 'weighted_f1': '0.7635 ± 0.0119', 'loss': '0.6326 ± 0.0328', 'auroc': '0.9896 ± 0.0013'}
- ZS bootstrap: {'acc': '0.7204 ± 0.0134', 'bacc': '0.6130 ± 0.0173', 'kappa': '0.6966 ± 0.0245', 'nw_kappa': '0.7240 ± 0.0192', 'weighted_f1': '0.7116 ± 0.0143', 'loss': '3.4200 ± 0.0030', 'auroc': '0.9823 ± 0.0018'}

## Retrieval — TCGA-OT slide retrieval (patient-disjoint)
- DB=8226  queries=1348  leaking patients dropped=0  disjoint asserted=True

| Metric | Value | Reference |
|---|---|---|
| Acc@3 | 0.8717 | **0.880** |
| MVAcc@3 | 0.7812 | **0.807** |

## Retrieval Tier-1 — training-free post-processing (negative result)

Three standard image-retrieval techniques applied to the **frozen** TITAN embeddings. Nothing is trained; only the search procedure changes. Hyperparameters are selected on **val** queries and reported once on **test** queries, against a fixed patient-disjoint database.

**None of them beat the raw-cosine baseline.** Selected configs and their test deltas (selection metric Acc@3):

| Technique | Selected config (on val) | Acc@3 | ΔAcc@3 | MVAcc@3 | ΔMVAcc@3 |
|---|---|---|---|---|---|
| _baseline (raw cosine)_ | — | 0.8717 | — | 0.7812 | — |
| Whitening | `pca-384` | 0.8724 | +0.0007 | 0.7826 | +0.0015 |
| k-reciprocal | `k1=10 k2=3 λ=1.0` | 0.8717 | +0.0000 | 0.7812 | +0.0000 |
| αQE | `k=3 α=2.0` | 0.8702 | -0.0015 | 0.7819 | +0.0007 |
| DBA | `k=5 α=2.0` | 0.8702 | -0.0015 | 0.7752 | -0.0059 |

Notes:
- k-reciprocal's val-optimal setting is **λ=1.0**, which by construction *is* the plain cosine ranking — the sweep chose "do not re-rank at all". (λ=1.0 reproducing the baseline exactly also serves as the implementation's sanity check.)
- Whitening's `pca-384` row is **not** a win: it *lost* to the baseline on val (−0.0012) and was only selected as the sweep's argmax. Its +0.0007 test Acc@3 is a single query out of 1348 — noise, not effect.
- Full whitening (`pcaw-*`, `zca`) actively hurts Acc@3, implying TITAN's high-variance directions carry discriminative signal rather than nuisance variation — the usual motivation for whitening does not apply here.
- Every technique showed a small **MVAcc@3 gain on val** (+0.010 to +0.013), which looked like a real trade (more label-homogeneous top-3 at the cost of Acc@3). Re-selecting on MVAcc@3 and re-testing showed those gains **do not transfer to test** — all four flip negative (whitening −0.0074, k-reciprocal −0.0015, αQE −0.0067, DBA −0.0037). They were val-set noise, not signal. Those runs are saved alongside as `*_sel-mvacc3.json`.

## Sub-tasks — harder patient-disjoint TCGA-OT sub-typing

| Task | #cls | test n | LP bacc | Ret Acc@1 | MVAcc@3 |
|---|---|---|---|---|---|
| NSCLC_LUAD_vs_LUSC | 2 | 278 | 0.9051 | 0.8993 | 0.9065 |
| RCC_3way | 3 | 94 | 0.8570 | 0.8830 | 0.9149 |
| BRCA_IDC_vs_ILC | 2 | 67 | 0.9048 | 0.9254 | 0.9403 |
| Sarcoma_4way | 4 | 27 | 0.5333 | 0.4074 | 0.3704 |
| Brain_GBM_vs_LGG | 2 | 43 | 0.8242 | 0.8605 | 0.8605 |

## Retrieval Tier-3 — LoRA fine-tuning of the TITAN slide encoder (BRACS ROIs)

Adapting the **slide-aggregation transformer itself** with LoRA, on an out-of-distribution,
patient-disjoint, labeled cohort where the frozen model has real headroom: BRACS breast lesions,
7 classes (N, PB, UDH, FEA, ADH, DCIS, IC), 4,539 ROIs. CONCHv1.5 (patch encoder) stays frozen;
only the 6-block slide transformer is trained. Pipeline: `download_bracs.py --set roi` →
`cache_bracs_patch_features.py` (one-time CONCH patch cache) → `finetune_bracs_lora.py`.
Protocol: DB = train (3,657), queries = val (312, selection) / test (570, reported). Loss =
Proxy-Anchor (7 class proxies); trainable = LoRA (r=8, α=16) on the 24 block Linears
(`attn.qkv, attn.proj, mlp.fc1, mlp.fc2` × 6), 589,824 params. fp32 (the TITAN RTX has no bf16,
and Proxy-Anchor's `exp(α·cos)` terms overflow fp16). Model selected on **val retrieval** (not
val loss); 3 seeds.

**Result: a real but modest win — LoRA beats the frozen baseline and every training-free /
head-only control.** Test set, 570 patient-disjoint queries:

| Method | trained? | Acc@1 | Acc@3 | MVAcc@3 |
|---|---|---|---|---|
| Frozen TITAN (fp16 = fp32 LoRA-off) | no | 0.5053 | 0.7158 | 0.5281 |
| mean-CONCH-kNN (naive mean-pool, no aggregation) | no | 0.5246 | 0.7456 | 0.5474 |
| Frozen-embedding linear map + Proxy-Anchor | yes (3s) | 0.5123 | 0.7421 | 0.5251 |
| mean-CONCH linear map + Proxy-Anchor | yes (3s) | 0.5023 | 0.7333 | 0.5339 |
| **Block-LoRA + Proxy-Anchor** | **yes (3s)** | **0.5368 ±0.015** | **0.7573 ±0.002** | **0.5632 ±0.009** |

- **Acc@3 is the robust signal**: +0.042 over baseline with seed std 0.002 (all seeds 0.754–0.760).
- **Acc@1**: +0.032 mean; mean−std (0.522) clears baseline. Paired bootstrap (per-query, same
  DB) vs the fp32 LoRA-off baseline: seed deltas +0.021 / +0.053 / +0.021; the **val-selected
  best seed is the significant one** (Δ+0.053, CI [+0.014, +0.089] excludes 0), the other two
  CIs include 0. So Acc@1 is positive-but-marginal per-seed, decisive only for the model you'd
  actually pick.
- **Beats the head-only controls** (frozen-map 0.512, mean-CONCH 0.525) on every metric →
  adapting the encoder adds value beyond a linear reprojection or mean-pool. The
  encoder-adaptation thesis survives this stress test.

**Where the gains come from (per-class Acc@1, baseline → LoRA mean):**
DCIS 0.494→0.600 (+0.106), N 0.543→0.617 (+0.074), PB 0.418→0.468 (+0.051); IC/FEA/ADH ≈ flat;
UDH 0.305→0.276 (−0.028). LoRA re-aggregates to help mid-difficulty classes where patch signal
exists, but does **not** crack the atypical ADH/UDH pair — consistent with the frozen confusion
matrix, where ADH/UDH/DCIS retrieve each other 56–80% of the time. That inseparability is
**CONCH-level** (the frozen patch features), which an aggregator LoRA cannot fix.

**Context — why this ROI result matters despite being modest.** The Step-0 diagnostic showed
naive mean-pooling *ties* TITAN's learned aggregator here (0.525 vs 0.505 Acc@1): on tiny ROIs
(33% are ≤2 patches) the aggregator has little to do, so the ceiling is low by construction.
That LoRA still beats mean-pool on this stress test is the encouraging part — on full gigapixel
WSIs, where the aggregator does the heavy lifting, the expected payoff is larger. Overfitting
appeared exactly as predicted (train loss falls past ~epoch 5 while val Acc@3 declines); early
stopping on val retrieval caught it every seed.

Artifacts: `results/finetune_bracs_lora_step3.json` (per-seed metrics + paired tests),
`results/finetune_bracs_lora_step0_{confusion,controls}.json`,
`results/finetune_bracs_lora_baseline_fp16.json`.
## Retrieval — BRACS ROI (patient-disjoint baseline)

Frozen TITAN, raw cosine. DB(train)=3657  val_q=312  test_q=570  7 classes.

| Metric | Test | Chance (random retrieval) | Chance (majority class) |
|---|---|---|---|
| Acc@1 | 0.5053 | 0.1431 | 0.1491 |
| Acc@3 | 0.7158 | — | — |
| MVAcc@3 | 0.5281 | — | — |

Per-class Acc@1: ADH=0.278, DCIS=0.494, FEA=0.675, IC=0.815, N=0.543, PB=0.418, UDH=0.305
