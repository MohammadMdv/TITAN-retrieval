#!/usr/bin/env bash
# Run TITAN validation phases 2->4 + results table. Live output (tqdm bars visible).
# Usage:  bash validation/run_all.sh   (activate your venv first, or set TITAN_PYTHON)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${TITAN_PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

run() {
  echo ""; echo "########## $1 ##########"
  $PY "$2"
  echo "----- exit: $? -----"
}

run "PHASE 2  (CAMELYON16)"  phase2_camelyon.py
run "PHASE 3  (TCGA-OT zero-shot + linear probe)"  phase3_tcga_ot.py
run "PHASE 4  (TCGA-OT retrieval)"  phase4_retrieval.py

echo ""; echo "########## RESULTS TABLE ##########"
$PY make_results.py
echo ""; echo "ALL DONE."
