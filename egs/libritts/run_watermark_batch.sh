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

# TraceableSpeech watermark settings
TS_ENABLE=${TS_ENABLE:-true}
TS_CKPT=${TS_CKPT:-/home/wu25/mrnas04home/projects/TraceableSpeech/save_model320/g_00150000}
TS_SAMPLE_NUM=${TS_SAMPLE_NUM:-5}
TS_BIT_NUM=${TS_BIT_NUM:-4}

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
  --ts-enable "${TS_ENABLE}" \
  --ts-checkpoint-file "${TS_CKPT}" \
  --ts-sample-num "${TS_SAMPLE_NUM}" \
  --ts-bit-num "${TS_BIT_NUM}" \
  --top-k -100 \
  --temperature 1.0

echo "Evaluating watermark quality against clean references..."
python3 batch_watermark_quality.py --dir "${OUT_DIR}" --sr 16000 --json "${OUT_DIR}/metrics.json"

echo "Done. Full metrics: ${OUT_DIR}/metrics.json"
