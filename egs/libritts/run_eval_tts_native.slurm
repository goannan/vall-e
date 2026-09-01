#!/bin/bash
#SBATCH --job-name=eval_tts_native
#SBATCH --nodes=1
#SBATCH -w mrcompute02,mrcompute03,compute07
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

CKPT="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/genkai_models/tts_native_step0020000_epoch001.pt"

echo "================================================================================"
echo " Starting Full Dual-Dataset Benchmark for [TTS-NATIVE (step 20000)] "
echo " Datasets: LibriTTS (2,078) & Seed-TTS (1,088)"
echo " Checkpoint: ${CKPT}"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Evaluate on LibriTTS
echo "[1/2] Running TTS-Native on LibriTTS (test-clean, 2,078 cuts)..."
python3 test_valle_native_watermark.py \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/libriTTS \
    --watermark-model "${CKPT}" \
    --output-dir exp/eval_libri_tts_native \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

# 2. Evaluate on SeedTTS
echo "[2/2] Running TTS-Native on SeedTTS (1,088 cuts)..."
python3 test_valle_native_watermark.py \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/seedTTS \
    --watermark-model "${CKPT}" \
    --output-dir exp/eval_seed_tts_native \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

echo "TTS-Native (step 20000) Full Benchmark Completed Successfully!"
