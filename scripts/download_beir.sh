#!/usr/bin/env bash
# Download and unpack one BEIR dataset into the SPTAR data directory.
#
# Usage: bash scripts/download_beir.sh <dataset_name>
# Example: bash scripts/download_beir.sh fiqa
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <dataset_name>" >&2
  echo "Example: $0 fiqa" >&2
  exit 1
fi

DATASET="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEIR_DIR="${ROOT_DIR}/third_party/SPTAR/dense_retrieval/datasets/raw/beir"

mkdir -p "${BEIR_DIR}"

echo "Downloading ${DATASET} ..."
wget "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${DATASET}.zip" -O "${DATASET}.zip"
unzip -q "${DATASET}.zip" -d "${BEIR_DIR}"
rm "${DATASET}.zip"

echo "Done. Dataset is ready at ${BEIR_DIR}/${DATASET}/"
