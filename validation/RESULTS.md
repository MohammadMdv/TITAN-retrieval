# TITAN Validation Results

Reference numbers from the TITAN Nature Medicine paper / repo README.

## Phase 2 — CAMELYON16 binary (tumor vs normal)
- 100 slides (60 normal / 40 tumor). _Caveat: 512px@40x features (CONCHv1.5 expects ~20x); not a TITAN-designed task._

| Setting | Balanced acc | AUROC |
|---|---|---|
| Zero-shot | 0.5083 | -1.0000 |
| Linear probe | 0.9028 | 0.9630 |
- LP bootstrap: {'acc': '0.9003 ± 0.0548', 'bacc': '0.9027 ± 0.0557', 'kappa': '0.7900 ± 0.1146', 'nw_kappa': '0.7900 ± 0.1146', 'weighted_f1': '0.9009 ± 0.0543', 'loss': '0.6070 ± 0.4296', 'auroc': '0.9633 ± 0.0394'}

## Phase 3 — TCGA-OT (46-class, full WSIs)
| Setting | Balanced acc | Reference | AUROC | Kappa |
|---|---|---|---|---|
| Linear probe | 0.7026 | **0.704** | 0.9896 | 0.8080 |
| Zero-shot | 0.6150 | — | 0.9823 | 0.6984 |
- LP bootstrap: {'acc': '0.7803 ± 0.0110', 'bacc': '0.7043 ± 0.0195', 'kappa': '0.8097 ± 0.0179', 'nw_kappa': '0.8188 ± 0.0140', 'weighted_f1': '0.7635 ± 0.0119', 'loss': '0.6326 ± 0.0328', 'auroc': '0.9896 ± 0.0013'}
- ZS bootstrap: {'acc': '0.7204 ± 0.0134', 'bacc': '0.6130 ± 0.0173', 'kappa': '0.6966 ± 0.0245', 'nw_kappa': '0.7240 ± 0.0192', 'weighted_f1': '0.7116 ± 0.0143', 'loss': '3.4200 ± 0.0030', 'auroc': '0.9823 ± 0.0018'}

## Phase 4 — TCGA-OT slide retrieval (patient-disjoint)
DB = train split, queries = test split, cosine similarity, K=3. _(Numbers recovered from the
`phase9_eval_before_after.REF` baseline / phase9b's baseline row — the original
`phase4_retrieval.json` from this run wasn't preserved; re-run `phase4_retrieval.py` to
regenerate it directly.)_

| Metric | Value | Reference |
|---|---|---|
| Acc@3 | 0.8717 | **0.880** |
| MVAcc@3 | 0.7812 | **0.807** |

## Phase 8/9/9b — SupCon projection-head fine-tuning on frozen TITAN embeddings (negative result)
TITAN stays frozen throughout; only a small projection head (768→768→768) is trained with SupCon (Khosla et al. 2020) on the same OncoTreeCode labels the linear probe uses. Question: does a label-aware nonlinear projection of the frozen embedding beat the raw embedding on TCGA-OT retrieval/LP? **Answer: no** — val SupCon loss overfits almost immediately, and retrieval degrades monotonically with more training while LP bacc stays roughly flat.

- Training: 50 epochs, P=16×K=8 (128 per batch), best val SupCon loss at epoch 4/50.

| Epoch | val SupCon loss | LP bacc | ΔLP | Acc@3 | ΔAcc@3 | MVAcc@3 | ΔMVAcc@3 |
|---|---|---|---|---|---|---|---|
| baseline (no head) | — | 0.7026 | — | 0.8717 | — | 0.7812 | — |
| 1 | 4.6318 | 0.6978 | -0.0049 | 0.8776 | +0.0059 | 0.7767 | -0.0045 |
| 3 | 4.6039 | 0.6774 | -0.0253 | 0.8687 | -0.0030 | 0.7708 | -0.0104 |
| 5 | 4.5979 | 0.7038 | +0.0011 | 0.8539 | -0.0178 | 0.7752 | -0.0059 |
| 7 | 4.6365 | 0.6755 | -0.0271 | 0.8442 | -0.0274 | 0.7715 | -0.0096 |
| 10 | 4.6545 | 0.6808 | -0.0218 | 0.8583 | -0.0134 | 0.7864 | +0.0052 |
| 15 | 4.7635 | 0.6863 | -0.0164 | 0.8516 | -0.0200 | 0.7945 | +0.0134 |
| 20 | 4.8032 | 0.6643 | -0.0383 | 0.8346 | -0.0371 | 0.7930 | +0.0119 |
| 30 | 4.9253 | 0.6672 | -0.0354 | 0.8131 | -0.0586 | 0.7789 | -0.0022 |
| 40 | 5.0612 | 0.6839 | -0.0188 | 0.8049 | -0.0668 | 0.7856 | +0.0045 |
| 50 | 5.1108 | 0.6704 | -0.0322 | 0.8004 | -0.0712 | 0.7819 | +0.0007 |
