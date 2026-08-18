#!/bin/bash
#SBATCH --job-name=valle_gen_data
#SBATCH --nodelist=compute[03,06-07],mrcompute[02-03]
#SBATCH --nodes=1-4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
mkdir -p logs data/tokenized_voicemark

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
export OMP_NUM_THREADS=4

VALLE_CKPT="${1:-exp/valle_voicemark/epoch-40.pt}"
INPUT_MANIFEST="${2:-data/tokenized_voicemark/cuts_train.jsonl.gz}"
OUTPUT_PREFIX="${3:-data/tokenized_voicemark/cuts_train_valle_native}"
OUTPUT_H5_PREFIX="${4:-data/tokenized_voicemark/libritts_valle_native_train}"

TOTAL_TASKS=${SLURM_NTASKS:-4}

echo "=========================================================="
echo " Starting VALL-E Native Token Elastic Multi-GPU Generation "
echo " Time:            $(date)                                 "
echo " Allocated Nodes: ${SLURM_NODELIST:-auto}                 "
echo " Total Tasks:     ${TOTAL_TASKS} (1 GPU per task)         "
echo " VALL-E Model:    ${VALLE_CKPT}                           "
echo " Input Manifest:  ${INPUT_MANIFEST}                       "
echo " Output Manifest: ${OUTPUT_PREFIX}.jsonl.gz                "
echo "=========================================================="

# srun 会自动把任务发到被分配的节点上，每个任务独占自己分配到的那张 GPU
srun --export=ALL bash -c '
  RANK=${SLURM_PROCID}
  WORLD_SIZE=${SLURM_NTASKS}

  echo "[Node $(hostname)] >> Spawning Worker Rank ${RANK}/${WORLD_SIZE} on assigned GPU..."
  
  python generate_valle_native_dataset.py \
    --valle-checkpoint "'"${VALLE_CKPT}"'" \
    --input-manifest "'"${INPUT_MANIFEST}"'" \
    --output-manifest "'"${OUTPUT_PREFIX}.jsonl.gz"'" \
    --output-h5 "'"${OUTPUT_H5_PREFIX}.h5"'" \
    --rank "${RANK}" \
    --world-size "${WORLD_SIZE}" \
    --device "cuda:0" > "logs/gen_rank_${RANK}_${SLURM_JOB_ID}.log" 2>&1
'

echo "=========================================================="
echo " All workers completed. Merging ${TOTAL_TASKS} shard manifests... "
echo "=========================================================="

python -c "
from lhotse import CutSet
from pathlib import Path
shards = [CutSet.from_file(f'${OUTPUT_PREFIX}_rank{r}.jsonl.gz') for r in range(${TOTAL_TASKS}) if Path(f'${OUTPUT_PREFIX}_rank{r}.jsonl.gz').exists()]
if shards:
    merged = sum(shards[1:], shards[0])
    merged.to_file('${OUTPUT_PREFIX}.jsonl.gz')
    print(f'Successfully merged {len(merged)} cuts across all {len(shards)} ranks into ${OUTPUT_PREFIX}.jsonl.gz!')
"

echo "Dataset generation finished successfully at $(date)!"
