#!/bin/bash
#SBATCH --job-name=tts_native_wm
#SBATCH --nodes=1-4
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
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
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0,eth

CONFIG_PATH=${1:-"${WORK_DIR}/config_tts_native.json"}

# Parse nodes and master endpoint dynamically
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

echo "=========================================================="
echo " Starting NeuMark Multi-Node Multi-GPU Training           "
echo " Time:            $(date)                                 "
echo " Master Endpoint: ${MASTER_ADDR}:${MASTER_PORT}           "
echo " Total Tasks:     ${SLURM_NTASKS:-8}                      "
echo " Nodes Allocated: ${SLURM_JOB_NODELIST}                   "
echo " WorkingDir:      ${WORK_DIR}                             "
echo " Config:          ${CONFIG_PATH}                          "
echo "=========================================================="

srun bash -c "
  export PATH=/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin:\$PATH
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  export NCCL_DEBUG=INFO
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME=eth0,eth
  export MASTER_ADDR=${MASTER_ADDR}
  export MASTER_PORT=${MASTER_PORT}
  export RANK=\$SLURM_PROCID
  export LOCAL_RANK=\$SLURM_LOCALID
  export WORLD_SIZE=\$SLURM_NTASKS

  python3 tts_native_train.py --config '${CONFIG_PATH}'
"

echo "=========================================================="
echo " Training finished at $(date)                             "
echo "=========================================================="
