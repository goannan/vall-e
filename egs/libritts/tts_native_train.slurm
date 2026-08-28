#!/bin/bash
#SBATCH --job-name=tts_native_wm
#SBATCH --nodes=2
#SBATCH --nodelist=compute07,mrcompute03
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
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
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0,eth

CONFIG_PATH=${1:-"${WORK_DIR}/config_tts_native.json"}

# Parse nodes and master endpoint
NODES_ARRAY=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_NODE=${NODES_ARRAY[0]}
MASTER_PORT=29500
NUM_NODES=${#NODES_ARRAY[@]}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

echo "=========================================================="
echo " Starting NeuMark Multi-Node Multi-GPU Training           "
echo " Time:            $(date)                                 "
echo " Master Node:     ${MASTER_NODE}:${MASTER_PORT}           "
echo " Total Nodes:     ${NUM_NODES} (${NODES_ARRAY[*]})       "
echo " GPUs Per Node:   ${GPUS_PER_NODE}                        "
echo " Total GPUs:      ${TOTAL_GPUS}                           "
echo " WorkingDir:      ${WORK_DIR}                             "
echo " Config:          ${CONFIG_PATH}                          "
echo "=========================================================="

if [ "${NUM_NODES}" -gt 1 ]; then
  echo "Launching Multi-Node Multi-GPU with srun & Accelerate..."
  srun bash -c "
    export PATH=/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin:\$PATH
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=eth0,eth
    
    accelerate launch \
      --multi_gpu \
      --num_machines ${NUM_NODES} \
      --num_processes ${TOTAL_GPUS} \
      --machine_rank \$SLURM_NODEID \
      --main_process_ip ${MASTER_NODE} \
      --main_process_port ${MASTER_PORT} \
      --same_network \
      --mixed_precision bf16 \
      --dynamo_backend no \
      tts_native_train.py --config '${CONFIG_PATH}'
  "
else
  echo "Launching Single-Node Multi-GPU Training (${GPUS_PER_NODE} GPUs)..."
  accelerate launch     --multi_gpu     --num_processes "${GPUS_PER_NODE}"     --mixed_precision bf16     --dynamo_backend no     tts_native_train.py --config "${CONFIG_PATH}"
fi

echo "=========================================================="
echo " Training finished at $(date)                             "
echo "=========================================================="
