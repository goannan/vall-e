#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs data/tokenized_voicemark

# Conda environment activation
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate voicemark || true
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export PYTHONPATH="${SCRIPT_DIR}/../../:${SCRIPT_DIR}/../../../icefall:${SCRIPT_DIR}/../../../NeuMark:${PYTHONPATH:-}"

VALLE_CKPT="${1:-exp/valle_voicemark/epoch-40.pt}"
INPUT_MANIFEST="${2:-data/tokenized_voicemark/cuts_train.jsonl.gz}"
OUTPUT_PREFIX="${3:-data/tokenized_voicemark/cuts_train_valle_native_v4}"
OUTPUT_H5_PREFIX="${4:-data/tokenized_voicemark/libritts_valle_native_train_v4}"
ST_CONFIG="${ST_CONFIG:-STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json}"
ST_CHECKPOINT="${ST_CHECKPOINT:-STmodels/pretrained_model/SpeechTokenizer.pt}"

# Auto-detect available GPU count & configure concurrent workers
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
# Each VALL-E model takes ~1.8GB VRAM; on 93GB GPUs we can easily run 6-8 workers per GPU concurrently
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
JOB_ID="${PJM_JOBID:-manual}"

echo "=========================================================="
echo " Starting VALL-E Native Token Generation on Genkai/PJM    "
echo " Host:            $(hostname)                             "
echo " Time:            $(date)                                 "
echo " GPU Count:       ${NUM_GPUS}                             "
echo " Workers/GPU:     ${WORKERS_PER_GPU} (Total ${TOTAL_WORKERS} Workers) "
echo " VALL-E Model:    ${VALLE_CKPT}                           "
echo " Prompt Tokens:   single-WAV SpeechTokenizer encode       "
echo " Input Manifest:  ${INPUT_MANIFEST}                       "
echo " Output Manifest: ${OUTPUT_PREFIX}.jsonl.gz                "
echo "=========================================================="

echo "Spawning ${TOTAL_WORKERS} parallel inference workers..."
PIDS=()
for r in $(seq 0 $((TOTAL_WORKERS - 1))); do
    GPU_ID=$((r % NUM_GPUS))
    LOG_FILE="logs/gen_rank_${r}_${JOB_ID}.log"
    echo " >> [Worker ${r}/${TOTAL_WORKERS}] Assigned to GPU cuda:${GPU_ID} -> ${LOG_FILE}"
    python3 generate_valle_native_dataset.py \
        --valle-checkpoint "${VALLE_CKPT}" \
        --input-manifest "${INPUT_MANIFEST}" \
        --output-manifest "${OUTPUT_PREFIX}.jsonl.gz" \
        --output-h5 "${OUTPUT_H5_PREFIX}.h5" \
        --st-config "${ST_CONFIG}" \
        --st-checkpoint "${ST_CHECKPOINT}" \
        --rank "${r}" \
        --world-size "${TOTAL_WORKERS}" \
        --device "cuda:${GPU_ID}" > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
done

echo "All ${TOTAL_WORKERS} workers launched. Waiting for completion..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "=========================================================="
echo " All workers completed. Merging ${TOTAL_WORKERS} shard manifests... "
echo "=========================================================="

python3 -c "
from lhotse import CutSet
from pathlib import Path
total_workers = ${TOTAL_WORKERS}
shards = [CutSet.from_file(f'${OUTPUT_PREFIX}_rank{r}.jsonl.gz') for r in range(total_workers) if Path(f'${OUTPUT_PREFIX}_rank{r}.jsonl.gz').exists()]
if shards:
    merged = sum(shards[1:], shards[0])
    merged.to_file('${OUTPUT_PREFIX}.jsonl.gz')
    print(f'Successfully merged {len(merged)} cuts across {len(shards)} shards into ${OUTPUT_PREFIX}.jsonl.gz!')
else:
    print('No rank shards found to merge.')
"

echo "Dataset generation finished successfully at $(date)!"
