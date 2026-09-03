#!/usr/bin/env bash
# Convert filtered pseudo queries into SPTAR/ColBERT training files.
#
# Usage: bash scripts/prepare_colbert_data.sh <dataset> <pseudo_json> <exp_name>
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 <dataset> <pseudo_json> <exp_name>" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
python -m igft.data.convert_colbert_data \
  --dataset "$1" \
  --pseudo "$2" \
  --exp-name "$3"
