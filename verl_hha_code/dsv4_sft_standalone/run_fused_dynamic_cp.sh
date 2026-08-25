#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/mnt/shared-storage-user/huanghaian/miniconda3}"
PYTHON="${CONDA_ROOT}/envs/dsv4_sft_bridge/bin/python"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/third_party/Megatron-LM:${PROJECT_DIR}/third_party/Megatron-Bridge/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${PROJECT_DIR}/train_dsv4_dynamic_cp_sft.py" \
  --num-layers "${NUM_LAYERS:-2}" \
  --max-seqlen-per-rank "${MAX_SEQLEN_PER_RANK:-128}" \
  --train-steps "${TRAIN_STEPS:-1}" \
  --fused-dsa \
  --dsa-indexer-loss-coeff "${DSA_INDEXER_LOSS_COEFF:-0.001}" \
  "$@"
