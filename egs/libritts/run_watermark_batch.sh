#!/bin/bash
set -euo pipefail

# Number of samples to generate. Override by setting N_SAMPLES env var.
N_SAMPLES=${N_SAMPLES:-1000}
# Output directory for generated wavs (clean + watermarked)
OUT_DIR=${OUT_DIR:-infer/wm_eval}

# Model/checkpoint settings (mirrors valid.sh defaults)
EXP_DIR=${EXP_DIR:-exp/valle}
CHECKPOINT=${CHECKPOINT:-${EXP_DIR}/best-valid-loss.pt}
TEXT_PROMPTS="This I read with great attention, while they sat silent."
AUDIO_PROMPTS="./prompts/8455_210777_000067_000000.wav"
TEXT_BASE="To get up and running quickly just follow the steps below."

# Watermark backend settings
WATERMARK_BACKEND=${WATERMARK_BACKEND:-voicemark}
TS_ENABLE=${TS_ENABLE:-false}
TS_CKPT=${TS_CKPT:-/home/wu25/mrnas04home/projects/TraceableSpeech/save_model320/g_00150000}
VOICEMARK_ROOT=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}
VOICEMARK_CHECKPOINT=${VOICEMARK_CHECKPOINT:-train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt}
OUTPUT_SR=${OUTPUT_SR:-16000}

# Build a | separated text string with N_SAMPLES entries.
TEXTS=$(N_SAMPLES=${N_SAMPLES} TEXT_BASE="${TEXT_BASE}" python - <<'PY'
import os
n = int(os.environ.get("N_SAMPLES", 1000))
base = os.environ.get("TEXT_BASE", "Sample")
print("|".join([f"{base} Sample {i}." for i in range(n)]))
PY
)

echo "Generating ${N_SAMPLES} samples to ${OUT_DIR} (clean + watermark)..."
python3 bin/infer.py \
  --output-dir "${OUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --text-prompts "${TEXT_PROMPTS}" \
  --audio-prompts "${AUDIO_PROMPTS}" \
  --text "${TEXTS}" \
  --watermark-backend "${WATERMARK_BACKEND}" \
  --ts-enable "${TS_ENABLE}" \
  --ts-checkpoint-file "${TS_CKPT}" \
  --voicemark-root "${VOICEMARK_ROOT}" \
  --voicemark-checkpoint "${VOICEMARK_CHECKPOINT}" \
  --top-k -100 \
  --temperature 1.0

echo "Evaluating watermark quality against clean references..."
python3 batch_watermark_quality.py --dir "${OUT_DIR}" --sr "${OUTPUT_SR}" --json "${OUT_DIR}/metrics.json"

echo "Done. Full metrics: ${OUT_DIR}/metrics.json"
