#!/bin/bash
#SBATCH --job-name=libri_ts_8gpu
#SBATCH --nodes=1-4
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=all
#SBATCH --time=12:00:00
#SBATCH --output=logs/libri_ts_8gpu_%j.out
#SBATCH --error=logs/libri_ts_8gpu_%j.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
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

TOTAL_TASKS=${SLURM_NTASKS:-8}
OUTPUT_BASE="exp/valle_traceablespeech_synthesis"
VALLE_CKPT="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt"
TS_CKPT="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000"
TS_CFG="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json"

echo "================================================================================"
echo " Starting Elastic Multi-Node 8-GPU LibriTTS Synthesis (2,078 cuts)"
echo " Nodes:        ${SLURM_NODELIST:-auto}"
echo " Tasks:        ${TOTAL_TASKS} (1 GPU per task)"
echo " VALL-E Model: ${VALLE_CKPT}"
echo " Date:         $(date)"
echo "================================================================================"

srun --export=ALL bash -c '
    RANK=${SLURM_PROCID}
    WORLD_SIZE=${SLURM_NTASKS}
    echo "[Node $(hostname)] Spawning Worker Rank ${RANK}/${WORLD_SIZE} on assigned GPU..."
    
    python3 synthesize_valle_traceablespeech_rank.py \
        --dataset libritts \
        --output-base-dir "exp/valle_traceablespeech_synthesis" \
        --valle-checkpoint "/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt" \
        --ts-checkpoint "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000" \
        --ts-config "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json" \
        --rank "${RANK}" \
        --world-size "${WORLD_SIZE}" \
        --device "cuda:0" > "logs/ts_synth_libri_multinode_rank_${RANK}_${SLURM_JOB_ID}.log" 2>&1
'

echo "Merging LibriTTS shards..."
python3 -c "
import json
from pathlib import Path
out_dir = Path('exp/valle_traceablespeech_synthesis/libritts')
records = []
for r in range(${TOTAL_TASKS}):
    p = out_dir / f'metadata_shard_{r:02d}_of_${TOTAL_TASKS:02d}.jsonl'
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
    --audio-dir "exp/valle_traceablespeech_synthesis/libritts" \
    --output-dir exp/eval_libri_traceablespeech_valle_native_8gpu \
    --ts-checkpoint "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000" \
    --ts-config "/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json" \
    --wavlm-checkpoint models/wavlm_large_finetune.pth \
    --device cuda:0

echo "================================================================================"
echo " Finished at $(date)!"
echo "================================================================================"
