#!/usr/bin/env bash
# Shared fixed-prompt Seed-TTS inference body for the two Slurm entry points.

set -euo pipefail

recipe_dir=/home/wu25/mrnas04home/projects/vall-e/egs/libritts
cd "${recipe_dir}"

export PYENV_ROOT=${PYENV_ROOT:-${HOME}/.pyenv}
export PATH="${PYENV_ROOT}/bin:${PATH}"
export PYENV_VERSION=valle
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    pyenv shell valle
fi

: "${EXP_DIR:?The backend wrapper must set EXP_DIR}"
: "${WATERMARK_BACKEND:?The backend wrapper must set WATERMARK_BACKEND}"
: "${OUT_DIR:?The backend wrapper must set OUT_DIR}"

case "${WATERMARK_BACKEND}" in
    voicemark)
        ts_enable=false
        ;;
    traceablespeech)
        ts_enable=true
        ;;
    *)
        echo "Unsupported WATERMARK_BACKEND=${WATERMARK_BACKEND}" >&2
        exit 1
        ;;
esac

ts_ckpt=${TS_CKPT:-../../traceableSpeech/g_00150000}
voicemark_root=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}
voicemark_config=${VOICEMARK_CONFIG:-STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json}
voicemark_st_checkpoint=${VOICEMARK_ST_CHECKPOINT:-STmodels/pretrained_model/SpeechTokenizer.pt}
voicemark_checkpoint=${VOICEMARK_CHECKPOINT:-/home/wu25/mrnas04home/projects/VoiceMark/checkpoints/ref_ori.pt}
seed_tts_root=${SEED_TTS_ROOT:-data/seed_tts_eval}
seed_tts_manifest=${SEED_TTS_MANIFEST:-${seed_tts_root}/en/meta.lst}
num_samples=${N_SAMPLES:-1088}
fixed_prompt_audio=${FIXED_PROMPT_AUDIO:-prompts/8455_210777_000067_000000.wav}
fixed_prompt_text=${FIXED_PROMPT_TEXT:-This I read with great attention, while they sat silent.}

mkdir -p logs

if [[ ! -f "${seed_tts_manifest}" ]]; then
    python3 prepare_seed_tts_eval.py --output-dir "${seed_tts_root}" --language en
fi

test -f "${EXP_DIR}/epoch-40.pt" || {
    echo "Missing VALL-E checkpoint: ${EXP_DIR}/epoch-40.pt" >&2
    exit 1
}
test -f "${fixed_prompt_audio}" || {
    echo "Missing fixed prompt audio: ${fixed_prompt_audio}" >&2
    exit 1
}
if [[ "${WATERMARK_BACKEND}" == "traceablespeech" ]]; then
    test -f "${ts_ckpt}" || {
        echo "Missing TraceableSpeech checkpoint: ${ts_ckpt}" >&2
        exit 1
    }
fi

echo "VALL-E checkpoint : ${EXP_DIR}/epoch-40.pt"
echo "Watermark backend : ${WATERMARK_BACKEND}"
if [[ "${WATERMARK_BACKEND}" == "voicemark" ]]; then
    echo "VoiceMark ST      : ${voicemark_st_checkpoint}"
    echo "VoiceMark WM      : ${voicemark_checkpoint}"
else
    echo "TraceableSpeech   : ${ts_ckpt}"
fi
echo "Seed-TTS targets  : ${seed_tts_manifest} (${num_samples})"
echo "Fixed prompt WAV  : ${fixed_prompt_audio}"
echo "Fixed prompt text : ${fixed_prompt_text}"
echo "Output directory  : ${OUT_DIR}"
echo "Python            : $(command -v python3)"

python3 bin/infer.py --output-dir "${OUT_DIR}" \
    --checkpoint "${EXP_DIR}/epoch-40.pt" \
    --seed-tts-manifest "${seed_tts_manifest}" \
    --seed-tts-num-samples "${num_samples}" \
    --seed-tts-primary-output watermarked \
    --seed-tts-fixed-prompt-audio "${fixed_prompt_audio}" \
    --seed-tts-fixed-prompt-text "${fixed_prompt_text}" \
    --watermark-backend "${WATERMARK_BACKEND}" \
    --ts-enable "${ts_enable}" \
    --ts-checkpoint-file "${ts_ckpt}" \
    --voicemark-root "${voicemark_root}" \
    --voicemark-config "${voicemark_config}" \
    --voicemark-st-checkpoint "${voicemark_st_checkpoint}" \
    --voicemark-checkpoint "${voicemark_checkpoint}" \
    --voicemark-embed-vq1 true \
    --top-k -100 \
    --temperature 1.0
