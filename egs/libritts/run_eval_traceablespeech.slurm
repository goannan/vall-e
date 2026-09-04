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

TS_CKPT="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000"
TS_CFG="/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json"

echo "================================================================================"
echo " Starting Full Triple-Dataset Benchmark for [TRACEABLESPEECH (VALL-E Native)] "
echo " Datasets: LibriTTS test-clean (2,078), LibriTTS test-other (2,391), Seed-TTS (1,088)"
echo " TraceableSpeech Vocoder: ${TS_CKPT}"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Evaluate on LibriTTS (test-clean, 2,078 cuts)
echo "[1/3] Running TraceableSpeech on LibriTTS (test-clean, 2,078 cuts)..."
python3 evaluate_traceablespeech_valle_native.py \
    --manifest data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz \
    --output-dir exp/eval_libri_traceablespeech \
    --ts-checkpoint "${TS_CKPT}" \
    --ts-config "${TS_CFG}" \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

# 2. Evaluate on LibriTTS (test-other, 2,391 cuts)
# echo "[2/3] Running TraceableSpeech on LibriTTS (test-other, 2,391 cuts)..."
# python3 evaluate_traceablespeech_valle_native.py \
#     --manifest data/tokenized_voicemark/cuts_test_other_valle_native.jsonl.gz \
#     --output-dir exp/eval_libri_other_traceablespeech \
#     --ts-checkpoint "${TS_CKPT}" \
#     --ts-config "${TS_CFG}" \
#     --num-samples -1 \
#     --save-audio-samples 20 \
#     --device cuda:0

# 3. Evaluate on SeedTTS (1,088 cuts)
echo "[3/3] Running TraceableSpeech on SeedTTS (1,088 cuts)..."
python3 evaluate_traceablespeech_valle_native.py \
    --manifest data/tokenized_voicemark/cuts_seedtts_valle_native.jsonl.gz \
    --output-dir exp/eval_seed_traceablespeech \
    --ts-checkpoint "${TS_CKPT}" \
    --ts-config "${TS_CFG}" \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

echo "================================================================================"
echo " TraceableSpeech Native VALL-E Full Triple-Dataset Benchmark Completed! "
echo "================================================================================"
