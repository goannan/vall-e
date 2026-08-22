#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs data/tokenized_voicemark

# 1. Conda environment activation
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate voicemark || true
fi

# 2. Environment Variables & Setup
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=4
export PYTHONPATH="${SCRIPT_DIR}/../../:${SCRIPT_DIR}/../../../icefall:${SCRIPT_DIR}/../../../NeuMark:${PYTHONPATH:-}"

VALLE_CKPT="${VALLE_CKPT:-exp/valle_voicemark/epoch-40.pt}"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config_tts_valle_native.json}"

# Auto-detect available GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
JOB_ID="${PJM_JOBID:-${SLURM_JOB_ID:-manual}}"

echo "=========================================================="
echo " Starting VALL-E Native Training Pipeline                 "
echo " Host:        $(hostname)                                 "
echo " Time:        $(date)                                     "
echo " Python:      $(which python3)                            "
echo " GPU Count:   ${NUM_GPUS}                                 "
echo " CUDA Device: ${CUDA_VISIBLE_DEVICES:-auto}               "
echo " Workers/GPU: ${WORKERS_PER_GPU} (Total ${TOTAL_WORKERS} Gen Workers) "
echo " Config:      ${CONFIG_PATH}                              "
echo "=========================================================="

# 3. Helper Function: Prepare Split if not already synthesized
prepare_split() {
    local SPLIT_NAME="$1"
    local INPUT_MF="data/tokenized_voicemark/cuts_${SPLIT_NAME}.jsonl.gz"
    local OUT_PREFIX="data/tokenized_voicemark/cuts_${SPLIT_NAME}_valle_native"
    local OUT_H5="data/tokenized_voicemark/libritts_valle_native_${SPLIT_NAME}"

    if [ -f "${OUT_PREFIX}.jsonl.gz" ] && [ -s "${OUT_PREFIX}.jsonl.gz" ]; then
        echo "[Prepare] Found existing ${OUT_PREFIX}.jsonl.gz, skipping generation."
        return 0
    fi

    echo "----------------------------------------------------------"
    echo "[Prepare] Generating VALL-E Tokens for [${SPLIT_NAME}]..."
    echo "  Input:   ${INPUT_MF}"
    echo "  Output:  ${OUT_PREFIX}.jsonl.gz"
    echo "  Workers: ${TOTAL_WORKERS} parallel inference streams"
    echo "----------------------------------------------------------"

    local PIDS=()
    for r in $(seq 0 $((TOTAL_WORKERS - 1))); do
        local GPU_ID=$((r % NUM_GPUS))
        local LOG_FILE="logs/gen_${SPLIT_NAME}_rank_${r}_${JOB_ID}.log"
        python3 generate_valle_native_dataset.py \
            --valle-checkpoint "${VALLE_CKPT}" \
            --input-manifest "${INPUT_MF}" \
            --output-manifest "${OUT_PREFIX}.jsonl.gz" \
            --output-h5 "${OUT_H5}.h5" \
            --rank "${r}" \
            --world-size "${TOTAL_WORKERS}" \
            --device "cuda:${GPU_ID}" > "${LOG_FILE}" 2>&1 &
        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    echo "[Prepare] Merging ${SPLIT_NAME} shards into ${OUT_PREFIX}.jsonl.gz..."
    python3 -c "
import glob, gzip
from pathlib import Path
shards = sorted(glob.glob('${OUT_PREFIX}*rank*.jsonl.gz'))
if shards:
    out_file = '${OUT_PREFIX}.jsonl.gz'
    count = 0
    with gzip.open(out_file, 'wt') as out_f:
        for sf in shards:
            with gzip.open(sf, 'rt') as in_f:
                for line in in_f:
                    out_f.write(line)
                    count += 1
    print(f'Successfully merged {count} cuts into {out_file}!')
else:
    print('No rank shards found to merge.')
"
    echo "[Prepare] Split [${SPLIT_NAME}] ready!"
}

# 4. Step 1: Pre-generate Dev, Test, Train if needed
prepare_split "dev"
prepare_split "test"
prepare_split "train"

# Update prompt audio mapping
echo "[Prepare] Updating prompt audio mappings..."
python3 -c "
import gzip, json
from pathlib import Path
out_p = Path('data/tokenized_voicemark/prompt_cuts_map.json')
mapping = {}
if out_p.exists():
    try:
        with open(out_p, 'r') as f:
            mapping = json.load(f)
    except Exception:
        mapping = {}
for name in ['cuts_train.jsonl.gz', 'cuts_dev.jsonl.gz', 'cuts_test.jsonl.gz']:
    p = Path('data/tokenized_voicemark') / name
    if p.exists():
        with gzip.open(p, 'rt') as f:
            for line in f:
                d = json.loads(line)
                cid = d['id']
                rec = d.get('recording')
                src = rec['sources'][0]['source'] if rec and rec.get('sources') else None
                if src:
                    mapping[cid] = (src, d.get('start', 0.0), d.get('duration', 0.0))
with open(out_p, 'w') as f:
    json.dump(mapping, f)
print(f'Prompt cuts map updated: {len(mapping)} total entries.')
"

# 5. Step 2: Start NeuMark Training
echo "=========================================================="
echo " Starting NeuMark Watermark Training                      "
echo " Config: ${CONFIG_PATH}                                   "
echo "=========================================================="

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "Launching Accelerate Multi-GPU (${NUM_GPUS} GPUs DDP)..."
  accelerate launch \
    --multi_gpu \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tts_native_train.py --config "${CONFIG_PATH}"
else
  echo "Launching Single-GPU Training (with Accelerate bf16)..."
  accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tts_native_train.py --config "${CONFIG_PATH}"
fi
