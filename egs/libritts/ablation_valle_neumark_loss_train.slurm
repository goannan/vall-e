#!/bin/bash
#SBATCH --job-name=ablation_valle_neumark_loss
#SBATCH --nodes=3
#SBATCH --nodelist=mrcompute02,mrcompute03,compute07
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Ensure working directory is the submission directory
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
mkdir -p logs

CONFIG_PATH=${1:-"${WORK_DIR}/config_ablation_valle_neumark_loss.json"}

# Parse nodes and master endpoint dynamically
NODES_ARRAY=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_NODE=${NODES_ARRAY[0]}
MASTER_PORT=29501
NUM_NODES=${#NODES_ARRAY[@]}
TOTAL_TASKS=${SLURM_NTASKS:-4}

echo "=========================================================="
echo " Starting Ablation Training (3 Nodes, Total 4 GPUs)        "
echo " Time:            $(date)                                 "
echo " Master Endpoint: ${MASTER_NODE}:${MASTER_PORT}           "
echo " Allocated Nodes: ${NUM_NODES} (${NODES_ARRAY[*]})       "
echo " Total Tasks/GPUs:${TOTAL_TASKS}                          "
echo " WorkingDir:      ${WORK_DIR}                             "
echo " Config:          ${CONFIG_PATH}                          "
echo "=========================================================="

srun bash -c "
  # Environment setup (PyEnv voicemark or Conda)
  if [ -d "/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin" ]; then
    export PATH="/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin:\$PATH"
  elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate voicemark || true
  fi

  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  export OMP_NUM_THREADS=4
  export NCCL_DEBUG=INFO
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME=^lo,docker0,eth2,eth3
  export GLOO_SOCKET_IFNAME=^lo,docker0,eth2,eth3

  # PyTorch / Accelerate DDP Environment Variables
  export WORLD_SIZE=\${SLURM_NTASKS:-4}
  export RANK=\${SLURM_PROCID:-0}
  export LOCAL_RANK=\${SLURM_LOCALID:-0}
  export MASTER_ADDR=${MASTER_NODE}
  export MASTER_PORT=${MASTER_PORT}

  python3 tts_native_train.py --config \"${CONFIG_PATH}\"
"

echo "=========================================================="
echo " Training finished at $(date)                             "
echo "=========================================================="
