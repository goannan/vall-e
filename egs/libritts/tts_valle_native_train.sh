#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Conda environment activation
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate voicemark || true
fi

# Auto-detect available GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)

echo "=========================================================="
echo " Starting VALL-E Native NeuMark Training                  "
echo " Host:        $(hostname)                                 "
echo " Time:        $(date)                                     "
echo " Python:      $(which python3)                            "
echo " GPU Count:   ${NUM_GPUS}                                 "
echo " CUDA Device: ${CUDA_VISIBLE_DEVICES:-auto}               "
echo "=========================================================="

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=4

CONFIG_PATH="${1:-${SCRIPT_DIR}/config_tts_valle_native.json}"

# Check and auto-generate Dev & Test VALL-E tokens if missing
if [ ! -f "data/tokenized_voicemark/cuts_dev_valle_native.jsonl.gz" ] || [ ! -f "data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz" ]; then
  echo "=========================================================="
  echo " Dev/Test VALL-E token manifests not found.               "
  echo " Auto-generating Dev & Test tokens before training...     "
  echo "=========================================================="
  bash "${SCRIPT_DIR}/generate_valle_native_dev_test.sh"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "Launching Accelerate Multi-GPU (${NUM_GPUS} GPUs DDP)..."
  accelerate launch \
    --multi_gpu \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tts_native_train.py --config "${CONFIG_PATH}"
else
  echo "Launching Single-GPU Training (with Accelerate bf16)..."
  accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tts_native_train.py --config "${CONFIG_PATH}"
fi
