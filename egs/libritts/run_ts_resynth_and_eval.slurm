#!/bin/bash
#SBATCH --job-name=ts_resynth_eval
#SBATCH --nodelist=mrcompute02,mrcompute03,compute07
#SBATCH --nodes=1-3
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=all
#SBATCH --time=12:00:00
#SBATCH --output=logs/ts_resynth_eval_%j.out
#SBATCH --error=logs/ts_resynth_eval_%j.err

set -euo pipefail
mkdir -p logs

export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export PYENV_ROOT="/home/wu25/mrnas04home/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell voicemark

PY_BIN="/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin/python3"
SCRIPT_DIR="/home/wu25/mrnas04home/projects/vall-e/egs/libritts"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo " Starting Full TraceableSpeech Re-Synthesis & Benchmark Evaluation at $(date)"
echo " Job ID: $SLURM_JOB_ID, Allocated Nodes: $SLURM_JOB_NODELIST, Tasks: $SLURM_NTASKS"
echo "================================================================================"

# -----------------------------------------------------------------------------
# STEP 1: Full 8-GPU Parallel Re-Synthesis (LibriTTS & SeedTTS)
# -----------------------------------------------------------------------------
LIBRI_SYNTH_DIR="exp/valle_traceablespeech_synthesis/libritts"
SEED_SYNTH_DIR="exp/valle_traceablespeech_synthesis/seedtts"
mkdir -p "$LIBRI_SYNTH_DIR" "$SEED_SYNTH_DIR"

echo ""
echo ">>> [Phase 1.1] 8-GPU Parallel Synthesis for LibriTTS (2,049 samples) <<<"
srun -n 8 bash -c '
    $0/bin/python3 synthesize_valle_traceablespeech_rank.py \
        --dataset libritts \
        --rank $SLURM_PROCID \
        --world-size $SLURM_NTASKS \
        --device cuda:0 \
        --output-dir exp/valle_traceablespeech_synthesis/libritts \
        --overwrite
' /home/wu25/mrnas04home/.pyenv/versions/voicemark

# Merge LibriTTS Metadata
$PY_BIN -c "
import glob, json
from pathlib import Path
out_dir = Path('exp/valle_traceablespeech_synthesis/libritts')
shards = sorted(glob.glob(str(out_dir / 'metadata_shard_*.jsonl')))
all_recs = []
for sf in shards:
    with open(sf) as f:
        for line in f:
            if line.strip():
                all_recs.append(json.loads(line))
with open(out_dir / 'metadata.jsonl', 'w') as f:
    for r in all_recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Merged {len(all_recs)} LibriTTS metadata records.')
"

echo ""
echo ">>> [Phase 1.2] 8-GPU Parallel Synthesis for SeedTTS (1,078 samples) <<<"
srun -n 8 bash -c '
    $0/bin/python3 synthesize_valle_traceablespeech_rank.py \
        --dataset seedtts \
        --rank $SLURM_PROCID \
        --world-size $SLURM_NTASKS \
        --device cuda:0 \
        --output-dir exp/valle_traceablespeech_synthesis/seedtts \
        --overwrite
' /home/wu25/mrnas04home/.pyenv/versions/voicemark

# Merge SeedTTS Metadata
$PY_BIN -c "
import glob, json
from pathlib import Path
out_dir = Path('exp/valle_traceablespeech_synthesis/seedtts')
shards = sorted(glob.glob(str(out_dir / 'metadata_shard_*.jsonl')))
all_recs = []
for sf in shards:
    with open(sf) as f:
        for line in f:
            if line.strip():
                all_recs.append(json.loads(line))
with open(out_dir / 'metadata.jsonl', 'w') as f:
    for r in all_recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Merged {len(all_recs)} SeedTTS metadata records.')
"

# -----------------------------------------------------------------------------
# STEP 2: Full 8-GPU Benchmark Evaluation (LibriTTS & SeedTTS)
# -----------------------------------------------------------------------------
LIBRI_EVAL_DIR="exp/eval_libri_traceablespeech_valle_native_8gpu"
SEED_EVAL_DIR="exp/eval_seed_traceablespeech_valle_native_8gpu"
rm -rf "$LIBRI_EVAL_DIR" "$SEED_EVAL_DIR"
mkdir -p "$LIBRI_EVAL_DIR" "$SEED_EVAL_DIR"

echo ""
echo ">>> [Phase 2.1] 8-GPU Parallel Benchmark Evaluation for LibriTTS <<<"
srun -n 8 bash -c '
    $0/bin/python3 evaluate_traceablespeech_shard.py \
        --audio-dir exp/valle_traceablespeech_synthesis/libritts \
        --output-dir exp/eval_libri_traceablespeech_valle_native_8gpu \
        --rank $SLURM_PROCID \
        --world-size $SLURM_NTASKS \
        --device cuda:0
' /home/wu25/mrnas04home/.pyenv/versions/voicemark

$PY_BIN merge_eval_shards.py \
    --eval-dir "$LIBRI_EVAL_DIR" \
    --dataset-name "LibriTTS-Test" \
    --manifest-path "synthesized_data/libriTTS/metadata.jsonl"

echo ""
echo ">>> [Phase 2.2] 8-GPU Parallel Benchmark Evaluation for SeedTTS <<<"
srun -n 8 bash -c '
    $0/bin/python3 evaluate_traceablespeech_shard.py \
        --audio-dir exp/valle_traceablespeech_synthesis/seedtts \
        --output-dir exp/eval_seed_traceablespeech_valle_native_8gpu \
        --rank $SLURM_PROCID \
        --world-size $SLURM_NTASKS \
        --device cuda:0
' /home/wu25/mrnas04home/.pyenv/versions/voicemark

$PY_BIN merge_eval_shards.py \
    --eval-dir "$SEED_EVAL_DIR" \
    --dataset-name "SeedTTS-En" \
    --manifest-path "synthesized_data/seedTTS/metadata.jsonl"

echo "================================================================================"
echo " All Synthesis and Evaluations Completed Successfully at $(date)"
echo "================================================================================"
