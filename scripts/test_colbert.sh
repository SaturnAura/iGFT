#!/usr/bin/env bash
# Index and evaluate a trained ColBERT retriever on the BEIR test split.
#
# All arguments are forwarded to SPTAR's test_colbert.sh, e.g.:
# bash scripts/test_colbert.sh -g 0 -d fiqa -e my_exp -p 60 -c 500
set -euo pipefail

SPTAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/SPTAR" && pwd)"
cd "${SPTAR_DIR}"
bash dense_retrieval/retriever/col_bert/test_colbert.sh "$@"
