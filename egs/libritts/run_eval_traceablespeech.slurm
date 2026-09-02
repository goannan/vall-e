#!/bin/bash
#SBATCH --job-name=eval_traceablespeech_native
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

VALLE_TS_CKPT="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/valle_checkpoints/valle_traceablespeech_epoch40.pt"
TS_CKPT="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000"
TS_CFG="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json"
FIXED_PROMPT_WAV="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/prompts/8455_210777_000067_000000.wav"

echo "================================================================================"
echo " Starting Full Dual-Dataset Benchmark for [TRACEABLESPEECH (VALL-E Native)] "
echo " VALL-E Checkpoint: ${VALLE_TS_CKPT}"
echo " TraceableSpeech Vocoder: ${TS_CKPT}"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Evaluate on LibriTTS Fixed Prompt Scenario (1,088 pairs from valle_traceablespeech_epoch40.pt)
echo "[1/2] Running TraceableSpeech on LibriTTS Prompt Pairs (1,088 samples)..."
python3 evaluate_traceablespeech_valle_native.py \
    --audio-dir infer/traceablespeech_seedtts_fixed_libritts_prompt_epoch40 \
    --output-dir exp/eval_libri_traceablespeech \
    --fixed-prompt-wav "${FIXED_PROMPT_WAV}" \
    --ts-checkpoint "${TS_CKPT}" \
    --ts-config "${TS_CFG}" \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

# 2. Evaluate on SeedTTS Dynamic Prompt Scenario (1,081 pairs from valle_traceablespeech_epoch40.pt)
echo "[2/2] Running TraceableSpeech on SeedTTS Dynamic Prompt Pairs (1,081 samples)..."
python3 evaluate_traceablespeech_valle_native.py \
    --audio-dir infer/traceablespeech_seedtts_en_epoch40 \
    --output-dir exp/eval_seed_traceablespeech \
    --prompt-dir synthesized_data/seedTTS/prompt \
    --ts-checkpoint "${TS_CKPT}" \
    --ts-config "${TS_CFG}" \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

echo "================================================================================"
echo " TraceableSpeech Native VALL-E Benchmark Completed Successfully! "
echo "================================================================================"
