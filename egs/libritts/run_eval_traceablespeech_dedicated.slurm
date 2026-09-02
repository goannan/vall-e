#!/bin/bash
#SBATCH --job-name=eval_ts_audio_quality
#SBATCH --nodes=1
#SBATCH -w mrcompute03,mrcompute02,compute07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

FIXED_PROMPT_WAV="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/prompts/8455_210777_000067_000000.wav"

echo "================================================================================"
echo " Starting VALL-E TraceableSpeech Audio Distortion & Perceptual Quality Benchmark "
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. LibriTTS Fixed Prompt VALL-E pairs (1,088 samples)
echo "[1/2] Evaluating LibriTTS Fixed Prompt VALL-E pairs (PESQ, STOI, UTMOS, SIM, WER)..."
python3 evaluate_traceablespeech_audio_quality_only.py \
    --audio-dir infer/traceablespeech_seedtts_fixed_libritts_prompt_epoch40 \
    --output-dir exp/eval_libri_traceablespeech_valle_pairs \
    --prompt-wav "${FIXED_PROMPT_WAV}" \
    --workers 16 \
    --device cuda:0

# 2. SeedTTS Dynamic Prompt VALL-E pairs (1,081 samples)
echo "[2/2] Evaluating SeedTTS Dynamic Prompt VALL-E pairs (PESQ, STOI, UTMOS, SIM, WER)..."
python3 evaluate_traceablespeech_audio_quality_only.py \
    --audio-dir infer/traceablespeech_seedtts_en_epoch40 \
    --output-dir exp/eval_seed_traceablespeech_valle_pairs \
    --workers 16 \
    --device cuda:0

echo "================================================================================"
echo " VALL-E TraceableSpeech Audio Quality Benchmark Completed Successfully! "
echo "================================================================================"
