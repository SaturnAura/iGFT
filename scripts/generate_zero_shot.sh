#!/usr/bin/env bash
# Generate pseudo queries with a zero-shot prompt (no SFT stage).
#
# Usage: bash scripts/generate_zero_shot.sh [cuda_device]
set -euo pipefail

CUDA_DEVICE="${1:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_DIR="${ROOT_DIR}/third_party/LLaMA-Factory"

cd "${LLM_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" llamafactory-cli generation "${ROOT_DIR}/configs/llamafactory/generation_zero_shot.yaml"
