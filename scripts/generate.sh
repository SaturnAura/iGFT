#!/usr/bin/env bash
# Generate pseudo queries with the SFT-tuned query generator (low-resource).
#
# Usage: bash scripts/generate.sh [cuda_device]
set -euo pipefail

CUDA_DEVICE="${1:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_DIR="${ROOT_DIR}/third_party/LLaMA-Factory"

cd "${LLM_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" llamafactory-cli generation "${ROOT_DIR}/configs/llamafactory/generation.yaml"
