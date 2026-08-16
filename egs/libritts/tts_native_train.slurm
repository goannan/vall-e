#!/bin/bash
#SBATCH --job-name=tts_native_wm
#SBATCH --nodes=1
#SBATCH --nodelist=compute[03,07],mrcompute[02-03]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Ensure working directory is the submission directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/home/wu25/mrnas04home/projects/vall-e/egs/libritts}"
cd "${WORK_DIR}"
mkdir -p logs

# Environment setup
PYENV_VERSION=${PYENV_VERSION:-valle}
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
export OMP_NUM_THREADS=4

CONFIG_PATH=${1:-"${WORK_DIR}/config_tts_native.json"}

echo "=========================================================="
echo " Starting TTS-Native Watermark 1-GPU Training             "
echo " Time:            $(date)                                 "
echo " Host:            $(hostname)                             "
echo " WorkingDir:      ${WORK_DIR}                             "
echo " CUDA Device:     ${CUDA_VISIBLE_DEVICES:-auto}           "
echo " Config:          ${CONFIG_PATH}                          "
echo "=========================================================="

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- Initial GPU Snapshot ---"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv
  echo "----------------------------"
fi

python3 tts_native_train.py --config "${CONFIG_PATH}"

echo "=========================================================="
echo " Training finished at $(date)                             "
echo "=========================================================="
