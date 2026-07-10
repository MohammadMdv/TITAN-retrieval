#!/usr/bin/env bash
# Run the TITAN validation suite + results table. Live output (tqdm bars visible).
# Usage:  bash validation/run_all.sh   (activate your venv first, or set TITAN_PYTHON)
#
# Skips smoke_slide_encoding / classification_camelyon16 unless CAMELYON16_ROOT is set,
# since those need local CAMELYON16 CONCHv1.5 features that are not part of this repo.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${TITAN_PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

run() {
  echo ""; echo "########## $1 ##########"
  $PY "$2"
  echo "----- exit: $? -----"
}

if [ -n "${CAMELYON16_ROOT:-}" ]; then
  run "CLASSIFICATION  (CAMELYON16)" classification_camelyon16.py
else
  echo "[skip] classification_camelyon16.py -- set CAMELYON16_ROOT to enable"
fi

run "CLASSIFICATION  (TCGA-OT zero-shot + linear probe)" classification_tcga_ot.py
run "RETRIEVAL       (TCGA-OT baseline, patient-disjoint)" retrieval_tcga_ot.py

# Tier-1: training-free retrieval post-processing on the frozen embeddings.
run "RETRIEVAL TIER-1 (whitening / PCA)" retrieval_tcga_ot_whitening.py
run "RETRIEVAL TIER-1 (k-reciprocal re-ranking)" retrieval_tcga_ot_kreciprocal.py
run "RETRIEVAL TIER-1 (query expansion + DBA)" retrieval_tcga_ot_query_expansion.py

echo ""; echo "########## RESULTS TABLE ##########"
$PY make_results.py
echo ""; echo "ALL DONE."
