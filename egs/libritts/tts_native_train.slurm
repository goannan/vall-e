#!/bin/bash
#SBATCH --job-name=tts_native_wm
#SBATCH --nodes=1
#SBATCH --nodelist=mrcompute02,mrcompute03,compute07
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Ensure working directory is the submission directory
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
mkdir -p logs

# Environment setup (PyEnv voicemark or Conda)
if [ -d "/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin" ]; then
  export PATH="/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin:$PATH"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate voicemark || true
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

CONFIG_PATH=${1:-"${WORK_DIR}/config_tts_native.json"}
TOTAL_GPUS=4

echo "=========================================================="
echo " Starting NeuMark Training (Single-Node 4 GPUs)           "
echo " Time:       $(date)                                      "
echo " Node:       $(hostname)                                  "
echo " Total GPUs: ${TOTAL_GPUS}                                "
echo " WorkingDir: ${WORK_DIR}                                  "
echo " Config:     ${CONFIG_PATH}                               "
echo "=========================================================="

accelerate launch \
  --multi_gpu \
  --num_processes "${TOTAL_GPUS}" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  tts_native_train.py --config "${CONFIG_PATH}"

echo "=========================================================="
echo " Training finished at $(date)                             "
echo "=========================================================="
