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
