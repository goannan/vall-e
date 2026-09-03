#!/bin/bash
#SBATCH --job-name=libri_full_2078
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --output=logs/libri_full_2078_%j.out
#SBATCH --error=logs/libri_full_2078_%j.err

echo "================================================================================"
echo " Starting Full 2,078 LibriTTS 8-GPU Synthesis with Completed Prompt Text"
echo " Date: $(date)"
echo " Host: $(hostname)"
echo " Job ID: $SLURM_JOB_ID"
echo " CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

mkdir -p logs exp/valle_traceablespeech_synthesis/libritts

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

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs on node $(hostname):"
nvidia-smi -L

OUTPUT_BASE="exp/valle_traceablespeech_synthesis"
VALLE_CKPT="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt"
TS_CKPT="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000"
TS_CFG="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json"

echo ""
echo ">>> Launching 8 Workers for Full LibriTTS (2,078 samples) <<<"
PIDS=()
for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Spawning LibriTTS Worker Rank ${RANK}/${NUM_GPUS} on physical GPU ${RANK}..."
    CUDA_VISIBLE_DEVICES=${RANK} python3 synthesize_valle_traceablespeech_rank.py \
        --dataset libritts \
        --output-base-dir "${OUTPUT_BASE}" \
        --valle-checkpoint "${VALLE_CKPT}" \
        --ts-checkpoint "${TS_CKPT}" \
        --ts-config "${TS_CFG}" \
        --rank "${RANK}" \
        --world-size "${NUM_GPUS}" \
        --device "cuda:0" > "logs/ts_synth_libri_full_rank_${RANK}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo "All 8 LibriTTS workers spawned. Waiting for completion..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "Merging LibriTTS shards..."
python3 -c "
import json
from pathlib import Path
out_dir = Path('${OUTPUT_BASE}/libritts')
records = []
for r in range(${NUM_GPUS}):
    p = out_dir / f'metadata_shard_{r:02d}_of_${NUM_GPUS:02d}.jsonl'
    if p.exists():
        with open(p, 'r') as f:
            for l in f:
                if l.strip():
                    records.append(json.loads(l))
records.sort(key=lambda x: x['utt_id'])
with open(out_dir / 'metadata.jsonl', 'w') as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
with open(out_dir / 'metadata.json', 'w') as f:
    json.dump({'dataset': 'libritts', 'total_records': len(records), 'records': records}, f, indent=2)
print(f'Successfully merged {len(records)} LibriTTS records into metadata.jsonl and metadata.json!')
"

echo "--- Evaluating Full LibriTTS (2,078 cuts) ---"
python3 evaluate_traceablespeech_valle_native.py \
    --audio-dir "${OUTPUT_BASE}/libritts" \
    --output-dir exp/eval_libri_traceablespeech_valle_native_8gpu \
    --ts-checkpoint "${TS_CKPT}" \
    --ts-config "${TS_CFG}" \
    --wavlm-checkpoint models/wavlm_large_finetune.pth \
    --device cuda:0

echo "================================================================================"
echo " Finished LibriTTS Full 2,078 Synthesis & Evaluation at $(date)!"
echo "================================================================================"
