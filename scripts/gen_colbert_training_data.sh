#!/usr/bin/env bash
# Generate ColBERT train/dev collections and triples from pseudo queries.
#
# Usage: bash scripts/gen_colbert_training_data.sh <dataset> <exp_name> [weak_num]
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <dataset> <exp_name> [weak_num]" >&2
  exit 1
fi

SPTAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/SPTAR" && pwd)"
cd "${SPTAR_DIR}"
python dense_retrieval/retriever/dpr/train/gen_data_for_colbert.py \
  --dataset_name "$1" \
  --exp_name "$2" \
  --weak_num "${3:-100k}"
