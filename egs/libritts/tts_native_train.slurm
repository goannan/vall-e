#!/bin/bash
#SBATCH --job-name=tts_native_wm
#SBATCH --nodes=1
#SBATCH --nodelist=compute[03,07],mrcompute[02-03]
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Ensure working directory is the submission directory
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
mkdir -p logs

# Environment setup
PYENV_VERSION=${PYENV_VERSION:-voicemark}
if [ "${PYENV_VERSION}" != "none" ]; then
  export PYENV_ROOT="${HOME}/mrnas04home/.pyenv"
  if [ ! -d "${PYENV_ROOT}" ]; then
    export PYENV_ROOT="${HOME}/.pyenv"
  fi
  if [ -d "${PYENV_ROOT}/bin" ]; then
    export PATH="${PYENV_ROOT}/bin:${PATH}"
  fi
  if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    pyenv activate "${PYENV_VERSION}" || true
  fi
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CONFIG_PATH=${1:-"${WORK_DIR}/config_tts_native.json"}

# Auto-detect available GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)

echo "=========================================================="
echo " Starting NeuMark Multi-GPU Training                      "
echo " Time:            $(date)                                 "
echo " Host:            $(hostname)                             "
echo " WorkingDir:      ${WORK_DIR}                             "
echo " GPU Count:       ${NUM_GPUS}                             "
echo " CUDA Device:     ${CUDA_VISIBLE_DEVICES:-auto}           "
echo " Config:          ${CONFIG_PATH}                          "
echo "=========================================================="

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- Initial GPU Snapshot ---"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
  echo "----------------------------"
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

echo "=========================================================="
echo " Training finished at $(date)                             "
echo "=========================================================="
