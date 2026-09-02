#!/bin/bash
#SBATCH --job-name=test_small_sample
#SBATCH --nodes=1
#SBATCH -w mrcompute03,mrcompute02,compute07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export PYENV_ROOT="/home/wu25/mrnas04home/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate voicemark

ST_CONFIG="/mrnas04/homes/wu25-1000089/projects/NeuMark/STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
ST_CKPT="/mrnas04/homes/wu25-1000089/projects/NeuMark/STmodels/pretrained_model/SpeechTokenizer.pt"
WAVLM_CKPT="/mrnas04/homes/wu25-1000089/projects/vall-e/egs/libritts/models/wavlm_large_finetune.pth"
TTS_NATIVE_CKPT="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/genkai_models/tts_native_latest.pt"
PROPOSED_CKPT="/home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/tts_native_neumark/20260829-003157/NeuMark_step_0050000_epoch_002.pt"

echo "================================================================================"
echo " Running 10-sample Test for TTS-Native..."
echo "================================================================================"
python3 test_valle_native_watermark.py \
  --manifest data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz \
  --output-dir exp/test_small_sample_tts_native \
  --st-config "$ST_CONFIG" \
  --st-checkpoint "$ST_CKPT" \
  --watermark-model "$TTS_NATIVE_CKPT" \
  --wavlm-checkpoint "$WAVLM_CKPT" \
  --num-samples 10 \
  --device cuda:0

echo "================================================================================"
echo " Running 10-sample Test for Proposed Native (50,000 steps)..."
echo "================================================================================"
python3 test_valle_native_watermark.py \
  --manifest data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz \
  --output-dir exp/test_small_sample_proposed \
  --st-config "$ST_CONFIG" \
  --st-checkpoint "$ST_CKPT" \
  --watermark-model "$PROPOSED_CKPT" \
  --wavlm-checkpoint "$WAVLM_CKPT" \
  --num-samples 10 \
  --device cuda:0
