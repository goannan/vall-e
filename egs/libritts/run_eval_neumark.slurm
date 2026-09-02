#!/bin/bash
#SBATCH --job-name=eval_neumark
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

echo "================================================================================"
echo " Starting Full Dual-Dataset Benchmark for [NEUMARK (VoiceMark ref_recon)] "
echo " Datasets: LibriTTS (2,078) & Seed-TTS (1,088)"
echo " Checkpoint: /home/wu25/mrnas04home/projects/VoiceMark/checkpoints/ref_recon.pt"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Evaluate on LibriTTS
echo "[1/2] Running NeuMark (ref_recon) on LibriTTS (test-clean, 2,078 cuts)..."
python3 test_valle_native_watermark.py \
    --manifest data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz \
    --watermark-model /home/wu25/mrnas04home/projects/VoiceMark/checkpoints/ref_recon.pt \
    --output-dir exp/eval_libri_neumark \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

# 2. Evaluate on SeedTTS
echo "[2/2] Running NeuMark (ref_recon) on SeedTTS (1,088 cuts)..."
python3 test_valle_native_watermark.py \
    --manifest data/tokenized_voicemark/cuts_seedtts_valle_native.jsonl.gz \
    --watermark-model /home/wu25/mrnas04home/projects/VoiceMark/checkpoints/ref_recon.pt \
    --output-dir exp/eval_seed_neumark \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

echo "NeuMark (ref_recon) Full Benchmark Completed Successfully!"
