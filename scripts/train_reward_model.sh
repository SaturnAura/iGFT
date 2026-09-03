#!/usr/bin/env bash
# Train the reward model used by the PPO stage.
#
# Usage: bash scripts/train_reward_model.sh [cuda_device]
set -euo pipefail

CUDA_DEVICE="${1:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_DIR="${ROOT_DIR}/third_party/LLaMA-Factory"

cd "${LLM_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" llamafactory-cli train "${ROOT_DIR}/configs/llamafactory/reward_model.yaml"
