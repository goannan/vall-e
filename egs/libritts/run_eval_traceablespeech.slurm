#!/bin/bash
#SBATCH --job-name=eval_traceablespeech
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


echo "================================================================================"
echo " Starting Full Dual-Dataset Benchmark for [TRACEABLESPEECH] "
echo " Datasets: LibriTTS (2,078) & Seed-TTS (1,088)"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Evaluate on LibriTTS
echo "[1/2] Running TraceableSpeech on LibriTTS (test-clean, 2,078 cuts)..."
python3 evaluate_external_watermark_test.py     --backend traceablespeech     --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/libriTTS     --output-dir exp/eval_libri_traceablespeech     --ts-checkpoint /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000     --ts-config /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json     --num-samples -1     --save-audio-samples 20     --device cuda:0

# 2. Evaluate on SeedTTS
echo "[2/2] Running TraceableSpeech on SeedTTS (1,088 cuts)..."
python3 evaluate_external_watermark_test.py     --backend traceablespeech     --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/seedTTS     --output-dir exp/eval_seed_traceablespeech     --ts-checkpoint /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000     --ts-config /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json     --num-samples -1     --save-audio-samples 20     --device cuda:0

echo "TraceableSpeech Full Benchmark Completed Successfully!"
