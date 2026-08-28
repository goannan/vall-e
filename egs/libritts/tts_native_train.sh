#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate voicemark environment (pyenv or conda)
if [ -d "/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin" ]; then
    export PATH="/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin:$PATH"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate voicemark || true
fi

# Auto-detect available GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)

echo "=========================================================="
echo " Starting NeuMark Training                                "
echo " Host:        $(hostname)                                 "
echo " Time:        $(date)                                     "
echo " Python:      $(which python3)                            "
echo " GPU Count:   ${NUM_GPUS}                                 "
echo " CUDA Device: ${CUDA_VISIBLE_DEVICES:-auto}               "
echo "=========================================================="

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

CONFIG_PATH="${1:-${SCRIPT_DIR}/config_tts_native.json}"

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "Launching Accelerate Multi-GPU (${NUM_GPUS} GPUs DDP)..."
  accelerate launch     --multi_gpu     --num_processes "${NUM_GPUS}"     --mixed_precision bf16     --dynamo_backend no     tts_native_train.py --config "${CONFIG_PATH}"
else
  echo "Launching Single-GPU Training (with Accelerate bf16)..."
  accelerate launch     --num_processes 1     --mixed_precision bf16     --dynamo_backend no     tts_native_train.py --config "${CONFIG_PATH}"
fi
