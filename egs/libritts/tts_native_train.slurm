#!/bin/bash
#SBATCH --job-name=tts_native_wm
#SBATCH --nodes=1-2
#SBATCH --nodelist=mrcompute03,compute07
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
export NCCL_SOCKET_IFNAME=^lo,docker0,eth2,eth3
export GLOO_SOCKET_IFNAME=^lo,docker0,eth2,eth3

CONFIG_PATH=${1:-"${WORK_DIR}/config_tts_native.json"}

# Parse nodes and master endpoint dynamically
NODES_ARRAY=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_NODE=${NODES_ARRAY[0]}
MASTER_PORT=29500
NUM_NODES=${#NODES_ARRAY[@]}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

echo "=========================================================="
echo " Starting NeuMark Training                                "
echo " Time:            $(date)                                 "
echo " Master Endpoint: ${MASTER_NODE}:${MASTER_PORT}           "
echo " Allocated Nodes: ${NUM_NODES} (${NODES_ARRAY[*]})       "
echo " GPUs Per Node:   ${GPUS_PER_NODE}                        "
echo " Total GPUs:      ${TOTAL_GPUS}                           "
echo " WorkingDir:      ${WORK_DIR}                             "
echo " Config:          ${CONFIG_PATH}                          "
echo "=========================================================="

if [ "${NUM_NODES}" -gt 1 ]; then
  echo "Launching Multi-Node Accelerate Training (${NUM_NODES} nodes, ${TOTAL_GPUS} total GPUs)..."
  srun bash -c "
    export PATH=/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin:\$PATH
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=^lo,docker0,eth2,eth3
    export GLOO_SOCKET_IFNAME=^lo,docker0,eth2,eth3
    
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
  echo "Launching Single-Node Multi-GPU Training (${TOTAL_GPUS} GPUs)..."
  accelerate launch     --multi_gpu     --num_processes "${TOTAL_GPUS}"     --mixed_precision bf16     --dynamo_backend no     tts_native_train.py --config "${CONFIG_PATH}"
fi

echo "=========================================================="
echo " Training finished at $(date)                             "
echo "=========================================================="
