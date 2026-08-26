#!/bin/bash
#SBATCH --job-name=valle_gen_data
#SBATCH --nodelist=compute[03,07],mrcompute[02-03]
#SBATCH --nodes=1-4
#SBATCH --ntasks=8
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
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

# Checkpoint model path (1st arg)
VALLE_CKPT="${1:-exp/valle_voicemark/epoch-40.pt}"

# Splits to process (2nd arg, default: "dev test train")
# Can also pass "all", or a space-separated list like "dev test", or a specific single manifest path
SPLITS_ARG="${2:-dev test train}"

if [ "${SPLITS_ARG}" = "all" ]; then
  SPLITS=("dev" "test" "train")
elif [[ "${SPLITS_ARG}" == *.jsonl* ]]; then
  # Backward compatibility: single custom manifest file passed directly
  SINGLE_INPUT="${SPLITS_ARG}"
  SINGLE_OUT_PREFIX="${3:-${SINGLE_INPUT%.jsonl*}}"
  SINGLE_OUT_H5="${4:-${SINGLE_OUT_PREFIX}}"
  SPLITS=("CUSTOM")
else
  read -ra SPLITS <<< "${SPLITS_ARG}"
fi

TOTAL_TASKS=${SLURM_NTASKS:-8}

echo "=========================================================="
echo " Starting VALL-E Native Token Elastic Multi-GPU Generation "
echo " Time:            $(date)                                 "
echo " Allocated Nodes: ${SLURM_NODELIST:-auto}                 "
echo " Total Tasks:     ${TOTAL_TASKS} (1 GPU per task)         "
echo " VALL-E Model:    ${VALLE_CKPT}                           "
echo " Target Splits:   ${SPLITS[*]}                            "
echo "=========================================================="

for SPLIT in "${SPLITS[@]}"; do
  if [ "${SPLIT}" = "CUSTOM" ]; then
    INPUT_MANIFEST="${SINGLE_INPUT}"
    OUTPUT_PREFIX="${SINGLE_OUT_PREFIX}"
    OUTPUT_H5_PREFIX="${SINGLE_OUT_H5}"
    SPLIT_NAME="custom"
  else
    INPUT_MANIFEST="data/tokenized_voicemark/cuts_${SPLIT}.jsonl.gz"
    OUTPUT_PREFIX="data/tokenized_voicemark/cuts_${SPLIT}_valle_native"
    OUTPUT_H5_PREFIX="data/tokenized_voicemark/libritts_valle_native_${SPLIT}"
    SPLIT_NAME="${SPLIT}"
  fi

  if [ ! -f "${INPUT_MANIFEST}" ]; then
    echo "[-] [Warning] Input manifest ${INPUT_MANIFEST} not found. Skipping split '${SPLIT_NAME}'."
    continue
  fi

  echo "----------------------------------------------------------"
  echo " Processing Split: [${SPLIT_NAME}]"
  echo " Input Manifest:   ${INPUT_MANIFEST}"
  echo " Output Manifest:  ${OUTPUT_PREFIX}.jsonl.gz"
  echo " Output H5 Prefix: ${OUTPUT_H5_PREFIX}.h5"
  echo " Start Time:       $(date)"
  echo "----------------------------------------------------------"

  export VALLE_CKPT_FOR_RUN="${VALLE_CKPT}"
  export INPUT_MANIFEST_FOR_RUN="${INPUT_MANIFEST}"
  export OUTPUT_MANIFEST_FOR_RUN="${OUTPUT_PREFIX}.jsonl.gz"
  export OUTPUT_H5_FOR_RUN="${OUTPUT_H5_PREFIX}.h5"
  export CURRENT_SPLIT_NAME="${SPLIT_NAME}"

  # srun 会自动把任务发到被分配的节点上，每个任务独占自己分配到的那张 GPU
  srun --export=ALL bash -c '
    RANK=${SLURM_PROCID}
    WORLD_SIZE=${SLURM_NTASKS}

    echo "[Node $(hostname)] >> Spawning Worker Rank ${RANK}/${WORLD_SIZE} for split [${CURRENT_SPLIT_NAME}] on assigned GPU..."
    
    python generate_valle_native_dataset.py \
      --valle-checkpoint "${VALLE_CKPT_FOR_RUN}" \
      --input-manifest "${INPUT_MANIFEST_FOR_RUN}" \
      --output-manifest "${OUTPUT_MANIFEST_FOR_RUN}" \
      --output-h5 "${OUTPUT_H5_FOR_RUN}" \
      --rank "${RANK}" \
      --world-size "${WORLD_SIZE}" \
      --device "cuda:0" > "logs/gen_${CURRENT_SPLIT_NAME}_rank_${RANK}_${SLURM_JOB_ID}.log" 2>&1
  '

  if [ "${TOTAL_TASKS}" -gt 1 ]; then
    echo "Merging ${TOTAL_TASKS} shard manifests for split '${SPLIT_NAME}'..."
    python -c "
from lhotse import CutSet
from pathlib import Path
shard_files = []
for r in range(${TOTAL_TASKS}):
    p1 = Path(f'${OUTPUT_PREFIX}_rank{r}.jsonl.gz')
    p2 = Path(f'${OUTPUT_PREFIX}.jsonl_rank{r}.jsonl.gz')
    if p1.exists():
        shard_files.append(p1)
    elif p2.exists():
        shard_files.append(p2)
shards = [CutSet.from_file(p) for p in shard_files]
if shards:
    merged = sum(shards[1:], shards[0])
    merged.to_file('${OUTPUT_PREFIX}.jsonl.gz')
    print(f'Successfully merged {len(merged)} cuts across all {len(shards)} ranks into ${OUTPUT_PREFIX}.jsonl.gz!')
"
  fi

  echo "[+] Finished split '${SPLIT_NAME}' at $(date)!"
  echo ""
done

echo "=========================================================="
echo " All specified datasets generated successfully at $(date)! "
echo "=========================================================="
