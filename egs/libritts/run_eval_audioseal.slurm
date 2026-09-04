#!/bin/bash
#SBATCH --job-name=eval_audioseal
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
echo " Starting Full Triple-Dataset Benchmark for [AUDIOSEAL] "
echo " Datasets: LibriTTS test-clean (2,078), LibriTTS test-other (2,391), Seed-TTS (1,088)"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Evaluate on LibriTTS (test-clean)
echo "[1/3] Running AudioSeal on LibriTTS (test-clean, 2,078 cuts)..."
python3 evaluate_external_watermark_test.py \
    --backend audioseal \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/libriTTS \
    --output-dir exp/eval_libri_audioseal \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

# 2. Evaluate on LibriTTS (test-other)
# echo "[2/3] Running AudioSeal on LibriTTS (test-other, 2,391 cuts)..."
# python3 evaluate_external_watermark_test.py \
#     --backend audioseal \
#     --manifest data/tokenized_voicemark/cuts_test_other_valle_native.jsonl.gz \
#     --output-dir exp/eval_libri_other_audioseal \
#     --num-samples -1 \
#     --save-audio-samples 20 \
#     --device cuda:0

# 3. Evaluate on SeedTTS
echo "[3/3] Running AudioSeal on SeedTTS (1,088 cuts)..."
python3 evaluate_external_watermark_test.py \
    --backend audioseal \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/seedTTS \
    --output-dir exp/eval_seed_audioseal \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

echo "================================================================================"
echo " AudioSeal Full Triple-Dataset Benchmark Completed Successfully! "
echo "================================================================================"
