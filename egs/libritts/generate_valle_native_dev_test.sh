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
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
JOB_ID="${PJM_JOBID:-manual}"

generate_split() {
    local SPLIT_NAME="$1"
    local INPUT_MF="data/tokenized_voicemark/cuts_${SPLIT_NAME}.jsonl.gz"
    local OUT_PREFIX="data/tokenized_voicemark/cuts_${SPLIT_NAME}_valle_native"
    local OUT_H5="data/tokenized_voicemark/libritts_valle_native_${SPLIT_NAME}"

    echo "=========================================================="
    echo " Generating VALL-E Tokens for ${SPLIT_NAME} Set           "
    echo " Input:  ${INPUT_MF}                                      "
    echo " Output: ${OUT_PREFIX}.jsonl.gz                           "
    echo " Workers: ${TOTAL_WORKERS}                                "
    echo "=========================================================="

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

    echo " >> Merging ${SPLIT_NAME} shards into ${OUT_PREFIX}.jsonl.gz..."
    python3 -c "
import gzip, glob
from pathlib import Path
shards = sorted(glob.glob('${OUT_PREFIX}_rank*.jsonl.gz'))
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
}

# 1. Generate Dev Set
generate_split "dev"

# 2. Generate Test Set
generate_split "test"

# 3. Update prompt cuts mapping
echo "Updating prompt cuts mapping..."
python3 -c "
import gzip, json
from pathlib import Path
out_p = Path('data/tokenized_voicemark/prompt_cuts_map.json')
mapping = {}
if out_p.exists():
    with open(out_p, 'r') as f:
        mapping = json.load(f)
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
print(f'Updated prompt cuts map: {len(mapping)} total entries.')
"

echo "Dev and Test dataset generation completed successfully at $(date)!"
