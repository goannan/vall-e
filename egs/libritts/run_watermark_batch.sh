#!/bin/bash
set -euo pipefail

# Seed-TTS-Eval English contains 1088 standard zero-shot TTS rows.
N_SAMPLES=${N_SAMPLES:-1088}
# Output directory for generated wavs (clean + watermarked)
OUT_DIR=${OUT_DIR:-infer/wm_eval_seedtts_en}

# Model/checkpoint settings (mirrors valid.sh defaults)
EXP_DIR=${EXP_DIR:-exp/valle_voicemark}
CHECKPOINT=${CHECKPOINT:-${EXP_DIR}/epoch-40.pt}
SEED_TTS_ROOT=${SEED_TTS_ROOT:-data/seed_tts_eval}
SEED_TTS_MANIFEST=${SEED_TTS_MANIFEST:-${SEED_TTS_ROOT}/en/meta.lst}

# Watermark backend settings
WATERMARK_BACKEND=${WATERMARK_BACKEND:-voicemark}
TS_ENABLE=${TS_ENABLE:-false}
TS_CKPT=${TS_CKPT:-/home/wu25/mrnas04home/projects/TraceableSpeech/save_model320/g_00150000}
VOICEMARK_ROOT=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}
VOICEMARK_CHECKPOINT=${VOICEMARK_CHECKPOINT:-train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt}
OUTPUT_SR=${OUTPUT_SR:-16000}

if [[ ! -f "${SEED_TTS_MANIFEST}" ]]; then
  python3 prepare_seed_tts_eval.py \
    --output-dir "${SEED_TTS_ROOT}" \
    --language en
fi

echo "Generating ${N_SAMPLES} Seed-TTS-Eval English samples to ${OUT_DIR}..."
python3 bin/infer.py \
  --output-dir "${OUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --seed-tts-manifest "${SEED_TTS_MANIFEST}" \
  --seed-tts-num-samples "${N_SAMPLES}" \
  --seed-tts-primary-output watermarked \
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
