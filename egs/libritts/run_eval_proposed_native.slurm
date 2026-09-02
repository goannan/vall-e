#!/bin/bash
#SBATCH --job-name=eval_proposed_native
#SBATCH --nodes=1
#SBATCH -w mrcompute03,mrcompute02,compute07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export PYENV_ROOT="/home/wu25/mrnas04home/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell voicemark


PROPOSED_CKPT=${PROPOSED_CKPT:-/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/tts_native_neumark/20260829-003157/NeuMark_step_0050000_epoch_002.pt}

echo "================================================================================"
echo " Starting Full Dual-Dataset Benchmark for [PROPOSED TTS-NATIVE WATERMARK] "
echo " Datasets: LibriTTS (2,078) & Seed-TTS (1,088)"
echo " Checkpoint: ${PROPOSED_CKPT}"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Evaluate on LibriTTS
echo "[1/2] Running Proposed Model on LibriTTS (test-clean, 2,078 cuts)..."
python3 test_valle_native_watermark.py \
    --manifest data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz \
    --watermark-model "${PROPOSED_CKPT}" \
    --output-dir exp/eval_libri_proposed \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

# 2. Evaluate on SeedTTS
echo "[2/2] Running Proposed Model on SeedTTS (1,088 cuts)..."
python3 test_valle_native_watermark.py \
    --manifest data/tokenized_voicemark/cuts_seedtts_valle_native.jsonl.gz \
    --watermark-model "${PROPOSED_CKPT}" \
    --output-dir exp/eval_seed_proposed \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

echo "Proposed TTS-Native Model Full Benchmark Completed Successfully!"
