#!/usr/bin/env bash
# Train a ColBERT dense retriever on the prepared (gold + pseudo) triples.
#
# All remaining arguments are forwarded to SPTAR's train_colbert.sh, e.g.:
# bash scripts/train_colbert.sh -g 0 -d fiqa -e my_exp -m 500 -s 500 -b 64
set -euo pipefail

SPTAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/SPTAR" && pwd)"
cd "${SPTAR_DIR}"
bash dense_retrieval/retriever/col_bert/train_colbert.sh "$@"
