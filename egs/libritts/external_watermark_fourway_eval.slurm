#!/usr/bin/env bash
#SBATCH --job-name=ext_wm_4way
#SBATCH --output=/home/wu25/mrnas04home/projects/vall-e/egs/libritts/logs/%x_%A_%a.out
#SBATCH --error=/home/wu25/mrnas04home/projects/vall-e/egs/libritts/logs/%x_%A_%a.err
#SBATCH --array=0-3%1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

# One sbatch command launches four cases:
#   0 AudioSeal + per-row Seed-TTS prompt clean speech
#   1 AudioSeal + fixed LibriTTS prompt clean speech
#   2 WavMark  + per-row Seed-TTS prompt clean speech
#   3 WavMark  + fixed LibriTTS prompt clean speech
#
# Default clean carriers come from the VoiceMark-aligned VALL-E checkpoint, so
# the same VoiceMark-generated clean utterance is presented to AudioSeal and
# WavMark. Override SEED_CLEAN_DIR/FIXED_CLEAN_DIR to use another clean set.

PROJECT_DIR=${PROJECT_DIR:-/home/wu25/mrnas04home/projects/vall-e/egs/libritts}
VALL_E_ROOT=${VALL_E_ROOT:-/home/wu25/mrnas04home/projects/vall-e}
AUDIOSEAL_ROOT=${AUDIOSEAL_ROOT:-/home/wu25/mrnas04home/projects/audioseal}
WAVMARK_ROOT=${WAVMARK_ROOT:-/home/wu25/mrnas04home/projects/wavmark}
SEED_CLEAN_DIR=${SEED_CLEAN_DIR:-${PROJECT_DIR}/infer/voicemark_seedtts_en_epoch40}
FIXED_CLEAN_DIR=${FIXED_CLEAN_DIR:-${PROJECT_DIR}/infer/voicemark_seedtts_fixed_libritts_prompt_epoch40}
FIXED_PROMPT_WAV=${FIXED_PROMPT_WAV:-${PROJECT_DIR}/prompts/8455_210777_000067_000000.wav}
FIXED_PROMPT_TEXT=${FIXED_PROMPT_TEXT:-This I read with great attention, while they sat silent.}
MANIFEST=${MANIFEST:-${PROJECT_DIR}/data/seed_tts_eval/en/meta.lst}
# Keep this separate from the earlier TraceableSpeech-carrier evaluation.
# Reusing that directory would incorrectly reuse embedded WAVs and attack caches.
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_DIR}/infer/external_watermark_fourway_from_voicemark}

EXTERNAL_PYENV=${EXTERNAL_PYENV:-audioseal}
AUTO_SETUP=${AUTO_SETUP:-true}
RUN_UTMOS_ONLY=${RUN_UTMOS_ONLY:-false}
RUN_EMBED=${RUN_EMBED:-true}
RUN_AUDIO_QUALITY=${RUN_AUDIO_QUALITY:-true}
RUN_UTMOS=${RUN_UTMOS:-true}
RUN_ROBUSTNESS=${RUN_ROBUSTNESS:-true}
RUN_WER_SIM=${RUN_WER_SIM:-true}
SKIP_CODEC_ATTACKS=${SKIP_CODEC_ATTACKS:-false}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-false}
LIMIT=${LIMIT:-}
MIN_DURATION=${MIN_DURATION:-1.0}
WHISPER_BATCH_SIZE=${WHISPER_BATCH_SIZE:-4}
UTMOS_REPO=${UTMOS_REPO:-tarepan/SpeechMOS:v1.2.0}
UTMOS_MODEL=${UTMOS_MODEL:-utmos22_strong}

TASK_ID=${SLURM_ARRAY_TASK_ID:-${CASE_ID:-0}}
case "${TASK_ID}" in
    0)
        BACKEND=audioseal
        PROMPT_MODE=seedtts_prompt
        SOURCE_DIR=${SEED_CLEAN_DIR}
        PROMPT_WAV=
        ;;
    1)
        BACKEND=audioseal
        PROMPT_MODE=libritts_prompt
        SOURCE_DIR=${FIXED_CLEAN_DIR}
        PROMPT_WAV=${FIXED_PROMPT_WAV}
        ;;
    2)
        BACKEND=wavmark
        PROMPT_MODE=seedtts_prompt
        SOURCE_DIR=${SEED_CLEAN_DIR}
        PROMPT_WAV=
        ;;
    3)
        BACKEND=wavmark
        PROMPT_MODE=libritts_prompt
        SOURCE_DIR=${FIXED_CLEAN_DIR}
        PROMPT_WAV=${FIXED_PROMPT_WAV}
        ;;
    *)
        echo "CASE_ID/SLURM_ARRAY_TASK_ID must be 0, 1, 2, or 3; got ${TASK_ID}" >&2
        exit 1
        ;;
esac

OUTPUT_DIR=${OUTPUT_ROOT}/${BACKEND}_${PROMPT_MODE}
RESULT_DIR=${OUTPUT_DIR}/seed_tts_metrics
mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}" "${RESULT_DIR}"

for variable in RUN_UTMOS_ONLY RUN_EMBED RUN_AUDIO_QUALITY RUN_UTMOS RUN_ROBUSTNESS RUN_WER_SIM SKIP_CODEC_ATTACKS FORCE_RECOMPUTE AUTO_SETUP; do
    value=${!variable}
    if [[ "${value}" != true && "${value}" != false ]]; then
        echo "${variable} must be true or false; got ${value}" >&2
        exit 1
    fi
done

test -f "${PROJECT_DIR}/external_watermark_eval.py" || {
    echo "Missing evaluator: ${PROJECT_DIR}/external_watermark_eval.py" >&2
    exit 1
}
test -d "${SOURCE_DIR}" || { echo "Missing clean source: ${SOURCE_DIR}" >&2; exit 1; }
test -f "${MANIFEST}" || { echo "Missing manifest: ${MANIFEST}" >&2; exit 1; }
if [[ -n "${PROMPT_WAV}" ]]; then
    test -f "${PROMPT_WAV}" || { echo "Missing fixed prompt: ${PROMPT_WAV}" >&2; exit 1; }
fi

export PYENV_ROOT=${PYENV_ROOT:-${HOME}/.pyenv}
export PATH="${PYENV_ROOT}/bin:${PATH}"
export PYENV_VERSION=${EXTERNAL_PYENV}
export PYTHONPATH="${AUDIOSEAL_ROOT}/src:${WAVMARK_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/wu25/mrnas04home/.cache/huggingface}
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    pyenv shell "${EXTERNAL_PYENV}"
fi

echo "========================================================================"
echo "Case ID          : ${TASK_ID}"
echo "Watermark model  : ${BACKEND}"
echo "Prompt condition : ${PROMPT_MODE}"
echo "Clean source     : ${SOURCE_DIR}"
echo "Output           : ${OUTPUT_DIR}"
echo "Python env       : ${EXTERNAL_PYENV} ($(command -v python3))"
echo "UTMOS-only mode  : ${RUN_UTMOS_ONLY}"
if [[ "${RUN_UTMOS_ONLY}" == true ]]; then
    echo "Modules          : UTMOS only (all other module switches are ignored)"
else
    echo "Modules          : embed=${RUN_EMBED}, quality=${RUN_AUDIO_QUALITY}, UTMOS=${RUN_UTMOS}, robustness=${RUN_ROBUSTNESS}, WER/SIM=${RUN_WER_SIM}"
fi
echo "========================================================================"

if [[ "${RUN_UTMOS_ONLY}" == true ]]; then
    test -f "${PROJECT_DIR}/batch_utmos.py" || {
        echo "Missing UTMOS-only evaluator: ${PROJECT_DIR}/batch_utmos.py" >&2
        exit 1
    }
    test -f "${PROJECT_DIR}/external_utmos_fourway_summary.py" || {
        echo "Missing UTMOS four-way summarizer: ${PROJECT_DIR}/external_utmos_fourway_summary.py" >&2
        exit 1
    }
    if ! python3 -c 'import torch, torchaudio, tqdm' >/dev/null 2>&1; then
        echo "Missing UTMOS dependencies in ${EXTERNAL_PYENV}." >&2
        exit 1
    fi

    utmos_cmd=(
        python3 "${PROJECT_DIR}/batch_utmos.py"
        --dir "${OUTPUT_DIR}"
        --json "${OUTPUT_DIR}/utmos_clean_wm.json"
        --repo "${UTMOS_REPO}"
        --model "${UTMOS_MODEL}"
        --device cuda:0
    )
    [[ "${FORCE_RECOMPUTE}" == true ]] && utmos_cmd+=(--force)
    "${utmos_cmd[@]}"

    summary_ready=true
    for case_dir in \
        audioseal_seedtts_prompt \
        audioseal_libritts_prompt \
        wavmark_seedtts_prompt \
        wavmark_libritts_prompt; do
        if [[ ! -s "${OUTPUT_ROOT}/${case_dir}/utmos_clean_wm.json" ]]; then
            summary_ready=false
            break
        fi
    done
    if [[ "${summary_ready}" == true ]]; then
        python3 "${PROJECT_DIR}/external_utmos_fourway_summary.py" \
            --root "${OUTPUT_ROOT}" \
            --fixed-prompt-audio "${FIXED_PROMPT_WAV}" \
            --fixed-prompt-text "${FIXED_PROMPT_TEXT}"
    else
        echo "Four-way summary waits until all four array cases have UTMOS JSON."
    fi
    exit 0
fi

if ! python3 -c 'import audioseal, julius, numpy, soundfile, torch, torchaudio, tqdm' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != true ]]; then
        echo "Missing external-watermark dependencies in ${EXTERNAL_PYENV}." >&2
        exit 1
    fi
    python3 -m pip install -e "${AUDIOSEAL_ROOT}" 'julius==0.2.7' 'soundfile==0.12.1' 'tqdm>=4.64'
fi
if ! PYTHONPATH="${WAVMARK_ROOT}/src:${PYTHONPATH}" python3 -c 'import wavmark' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != true ]]; then
        echo "WavMark cannot be imported from ${WAVMARK_ROOT}." >&2
        exit 1
    fi
    python3 -m pip install -e "${WAVMARK_ROOT}"
fi
if [[ "${RUN_AUDIO_QUALITY}" == true ]] && \
   ! python3 -c 'import librosa, pesq, pystoi' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != true ]]; then
        echo "Missing PESQ/STOI dependencies." >&2
        exit 1
    fi
    python3 -m pip install 'librosa==0.10.2.post1' 'pesq==0.0.4' 'pystoi==0.4.1'
fi
if [[ "${RUN_ROBUSTNESS}" == true && "${SKIP_CODEC_ATTACKS}" != true ]] && \
   ! python3 -c 'import dac, encodec, snac' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != true ]]; then
        echo "Missing Encodec/DAC/SNAC dependencies." >&2
        exit 1
    fi
    python3 -m pip install \
        'encodec==0.1.1' 'descript-audio-codec==1.0.0' 'snac==1.2.1'
fi

stages=()
[[ "${RUN_EMBED}" == true ]] && stages+=(embed)
[[ "${RUN_AUDIO_QUALITY}" == true ]] && stages+=(quality)
[[ "${RUN_ROBUSTNESS}" == true ]] && stages+=(robustness)
if (( ${#stages[@]} > 0 )); then
    STAGE_CSV=$(IFS=,; echo "${stages[*]}")
    external_cmd=(
        python3 "${PROJECT_DIR}/external_watermark_eval.py"
        --backend "${BACKEND}"
        --source-dir "${SOURCE_DIR}"
        --output-dir "${OUTPUT_DIR}"
        --stages "${STAGE_CSV}"
        --min-duration "${MIN_DURATION}"
        --vall-e-root "${VALL_E_ROOT}"
        --wavmark-root "${WAVMARK_ROOT}"
        --utmos-repo "${UTMOS_REPO}"
        --utmos-model "${UTMOS_MODEL}"
    )
    [[ -n "${LIMIT}" ]] && external_cmd+=(--limit "${LIMIT}")
    [[ "${SKIP_CODEC_ATTACKS}" == true ]] && external_cmd+=(--skip-codecs)
    [[ "${RUN_UTMOS}" != true ]] && external_cmd+=(--skip-utmos)
    [[ "${FORCE_RECOMPUTE}" == true ]] && external_cmd+=(--force)
    "${external_cmd[@]}"
fi

if [[ "${RUN_WER_SIM}" == true ]]; then
    metric_env=(
        PROJECT_DIR="${PROJECT_DIR}"
        AUDIO_DIR="${OUTPUT_DIR}"
        RESULT_DIR="${RESULT_DIR}"
        MANIFEST="${MANIFEST}"
        PROMPT_WAV="${PROMPT_WAV}"
        RUN_WATERMARK_QUALITY=false
        RUN_WER_SIM=true
        RUN_UTMOS=false
        STRICT_ALL_PAIRS=true
        FORCE_RECOMPUTE="${FORCE_RECOMPUTE}"
        WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE}"
    )
    env "${metric_env[@]}" bash "${PROJECT_DIR}/seed_tts_clean_wm_metrics.slurm"
fi

# Rebuild the per-case report with external watermark robustness and quality
# included beside WER and all three requested SIM values.
summary_cmd=(
    python3 "${PROJECT_DIR}/seed_tts_metric_utils.py" summarize
    --output-dir "${RESULT_DIR}"
    --audio-dir "${OUTPUT_DIR}"
)
if [[ "${RUN_WER_SIM}" != true ]]; then
    summary_cmd+=(--skip-wer-sim)
fi
if [[ -s "${OUTPUT_DIR}/audio_quality_clean_vs_wm.json" ]]; then
    summary_cmd+=(--audio-quality-json "${OUTPUT_DIR}/audio_quality_clean_vs_wm.json")
fi
if [[ -s "${OUTPUT_DIR}/watermark_summary.json" && -s "${OUTPUT_DIR}/watermark_validation_table.txt" ]]; then
    summary_cmd+=(
        --watermark-summary-json "${OUTPUT_DIR}/watermark_summary.json"
        --watermark-table "${OUTPUT_DIR}/watermark_validation_table.txt"
    )
fi
"${summary_cmd[@]}"
cp "${RESULT_DIR}/seed_tts_clean_wm_metrics.txt" "${RESULT_DIR}/all_metrics_summary.txt"
cp "${RESULT_DIR}/seed_tts_clean_wm_metrics.json" "${RESULT_DIR}/all_metrics_summary.json"

echo "Finished case ${TASK_ID}: ${BACKEND} / ${PROMPT_MODE}"
echo "Result table: ${RESULT_DIR}/all_metrics_summary.txt"
echo "Result JSON : ${RESULT_DIR}/all_metrics_summary.json"
