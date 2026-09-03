#!/usr/bin/env bash
# Build the filtered/reduced corpora that SPTAR training data generation needs.
#
# Usage: bash scripts/prepare_sptar_corpus.sh <dataset> [weak_num]
# Example: bash scripts/prepare_sptar_corpus.sh fiqa 100k
set -euo pipefail

DATASET="${1:-fiqa}"
WEAK_NUM="${2:-100k}"
SPTAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/SPTAR" && pwd)"

cd "${SPTAR_DIR}"
python dense_retrieval/data_process.py --dataset "${DATASET}" --weak-num "${WEAK_NUM}"
