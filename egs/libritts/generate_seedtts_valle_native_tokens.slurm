#!/bin/bash
#SBATCH --job-name=seedtts_gen_tokens
#SBATCH --nodes=1
#SBATCH -w 
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
mkdir -p logs data/tokenized_voicemark

export PYENV_ROOT="/home/wu25/mrnas04home/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell voicemark

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

VALLE_CKPT="${1:-valle_checkpoints/valle_voicemark_epoch40.pt}"
META_LST="data/seed_tts_eval/en/meta.lst"
PROMPT_WAV_DIR="data/seed_tts_eval/en"
OUTPUT_PREFIX="data/tokenized_voicemark/cuts_seedtts_valle_native"
OUTPUT_H5_PREFIX="data/tokenized_voicemark/seedtts_valle_native"
NUM_GPUS=8

echo "=========================================================="
echo " Starting SeedTTS VALL-E Native 8-GPU Token Generation    "
echo " Time:            $(date)                                 "
echo " Node:            $(hostname)                             "
echo " Total GPUs:      ${NUM_GPUS}                             "
echo " VALL-E Model:    ${VALLE_CKPT}                           "
echo " Meta LST:        ${META_LST} (1,088 samples)             "
echo " Output Manifest: ${OUTPUT_PREFIX}.jsonl.gz               "
echo " Output H5:       ${OUTPUT_H5_PREFIX}.h5                  "
echo "=========================================================="

# Launch 8 worker processes in parallel across 8 GPUs
PIDS=()
for RANK in $(seq 0 $((NUM_GPUS - 1))); do
  echo "Spawning Worker Rank ${RANK}/${NUM_GPUS} on GPU cuda:${RANK}..."
  CUDA_VISIBLE_DEVICES=${RANK} python3 generate_seedtts_valle_native_tokens.py \
    --valle-checkpoint "${VALLE_CKPT}" \
    --meta-lst "${META_LST}" \
    --prompt-wav-dir "${PROMPT_WAV_DIR}" \
    --output-manifest "${OUTPUT_PREFIX}.jsonl.gz" \
    --output-h5 "${OUTPUT_H5_PREFIX}.h5" \
    --rank "${RANK}" \
    --world-size "${NUM_GPUS}" \
    --device "cuda:0" > "logs/seedtts_gen_rank_${RANK}_${SLURM_JOB_ID}.log" 2>&1 &
  PIDS+=($!)
done

echo "All 8 workers spawned. Waiting for completion..."
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "Merging ${NUM_GPUS} rank manifests into single CutSet ${OUTPUT_PREFIX}.jsonl.gz..."
python3 -c "
from lhotse import CutSet
from pathlib import Path
shard_files = []
for r in range(${NUM_GPUS}):
    p = Path(f'${OUTPUT_PREFIX}_rank{r}.jsonl.gz')
    if p.exists():
        shard_files.append(p)
shards = [CutSet.from_file(p) for p in shard_files]
if shards:
    merged = sum(shards[1:], shards[0])
    merged.to_file('${OUTPUT_PREFIX}.jsonl.gz')
    print(f'Successfully merged {len(merged)} SeedTTS cuts across all {len(shards)} ranks into ${OUTPUT_PREFIX}.jsonl.gz!')
"

echo "=========================================================="
echo " SeedTTS VALL-E Native Token Generation Completed: $(date)"
echo "=========================================================="
