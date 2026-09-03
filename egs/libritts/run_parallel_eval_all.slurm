#!/bin/bash
#SBATCH --job-name=ts_eval_8gpu
#SBATCH --nodes=1-4
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --output=logs/ts_eval_8gpu_%j.out
#SBATCH --error=logs/ts_eval_8gpu_%j.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
mkdir -p logs exp/eval_libri_traceablespeech_valle_native_8gpu exp/eval_seed_traceablespeech_valle_native_8gpu

export PYENV_ROOT="/home/wu25/mrnas04home/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell voicemark

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

TOTAL_TASKS=${SLURM_NTASKS:-8}
TS_CKPT="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000"
TS_CFG="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json"
WAVLM_CKPT="models/wavlm_large_finetune.pth"

echo "================================================================================"
echo " Starting Parallel 8-GPU TraceableSpeech Benchmark Evaluation"
echo " Date: $(date)"
echo " Tasks: ${TOTAL_TASKS}"
echo "================================================================================"

echo ""
echo ">>> Phase 1: Evaluating LibriTTS (2,049 samples) on 8 GPUs <<<"
srun --export=ALL bash -c '
    RANK=${SLURM_PROCID}
    WORLD_SIZE=${SLURM_NTASKS}
    python3 evaluate_traceablespeech_shard.py \
        --audio-dir exp/valle_traceablespeech_synthesis/libritts \
        --output-dir exp/eval_libri_traceablespeech_valle_native_8gpu \
        --rank "${RANK}" \
        --world-size "${WORLD_SIZE}" \
        --ts-checkpoint "'"${TS_CKPT}"'" \
        --ts-config "'"${TS_CFG}"'" \
        --wavlm-checkpoint "'"${WAVLM_CKPT}"'" \
        --device "cuda:0" > "logs/eval_libri_shard_${RANK}_${SLURM_JOB_ID}.log" 2>&1
'

echo "Merging LibriTTS Evaluation..."
python3 merge_eval_shards.py \
    --eval-dir exp/eval_libri_traceablespeech_valle_native_8gpu \
    --dataset-name "LibriTTS-Test"

echo ""
echo ">>> Phase 2: Evaluating SeedTTS (1,078 samples) on 8 GPUs <<<"
srun --export=ALL bash -c '
    RANK=${SLURM_PROCID}
    WORLD_SIZE=${SLURM_NTASKS}
    python3 evaluate_traceablespeech_shard.py \
        --audio-dir exp/valle_traceablespeech_synthesis/seedtts \
        --output-dir exp/eval_seed_traceablespeech_valle_native_8gpu \
        --rank "${RANK}" \
        --world-size "${WORLD_SIZE}" \
        --ts-checkpoint "'"${TS_CKPT}"'" \
        --ts-config "'"${TS_CFG}"'" \
        --wavlm-checkpoint "'"${WAVLM_CKPT}"'" \
        --device "cuda:0" > "logs/eval_seed_shard_${RANK}_${SLURM_JOB_ID}.log" 2>&1
'

echo "Merging SeedTTS Evaluation..."
python3 merge_eval_shards.py \
    --eval-dir exp/eval_seed_traceablespeech_valle_native_8gpu \
    --dataset-name "SeedTTS-En"

echo "================================================================================"
echo " All Evaluations Finished Successfully at $(date)!"
echo "================================================================================"
