#!/usr/bin/env bash
#SBATCH --job-name=wm_runtime
#SBATCH --output=/home/wu25/mrnas04home/projects/vall-e/egs/libritts/logs/%x_%j.out
#SBATCH --error=/home/wu25/mrnas04home/projects/vall-e/egs/libritts/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/wu25/mrnas04home/projects/vall-e/egs/libritts}
VALL_E_ROOT=${VALL_E_ROOT:-/home/wu25/mrnas04home/projects/vall-e}
AUDIOSEAL_ROOT=${AUDIOSEAL_ROOT:-/home/wu25/mrnas04home/projects/audioseal}
WAVMARK_ROOT=${WAVMARK_ROOT:-/home/wu25/mrnas04home/projects/wavmark}
VOICEMARK_ROOT=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}

N_SAMPLES=${N_SAMPLES:-1088}
MANIFEST=${MANIFEST:-${PROJECT_DIR}/data/seed_tts_eval/en/meta.lst}
FIXED_PROMPT_AUDIO=${FIXED_PROMPT_AUDIO:-${PROJECT_DIR}/prompts/8455_210777_000067_000000.wav}
FIXED_PROMPT_TEXT=${FIXED_PROMPT_TEXT:-This I read with great attention, while they sat silent.}

VM_VALLE_CHECKPOINT=${VM_VALLE_CHECKPOINT:-${PROJECT_DIR}/exp/valle_voicemark/epoch-40.pt}
TS_VALLE_CHECKPOINT=${TS_VALLE_CHECKPOINT:-${PROJECT_DIR}/exp/valle/epoch-40.pt}
VOICEMARK_CONFIG=${VOICEMARK_CONFIG:-STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json}
VOICEMARK_ST_CHECKPOINT=${VOICEMARK_ST_CHECKPOINT:-STmodels/pretrained_model/SpeechTokenizer.pt}
VOICEMARK_CHECKPOINT=${VOICEMARK_CHECKPOINT:-${VOICEMARK_ROOT}/checkpoints/ref_ori.pt}
TS_CHECKPOINT=${TS_CHECKPOINT:-${VALL_E_ROOT}/traceableSpeech/g_00150000}

RUN_NAME=${RUN_NAME:-fixed_prompt_n${N_SAMPLES}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_DIR}/infer/watermark_runtime_benchmark/${RUN_NAME}}
VM_OUTPUT_DIR=${VM_OUTPUT_DIR:-${OUTPUT_ROOT}/voicemark}
TS_OUTPUT_DIR=${TS_OUTPUT_DIR:-${OUTPUT_ROOT}/traceablespeech}
VM_TIMING_JSON=${VM_TIMING_JSON:-${OUTPUT_ROOT}/voicemark_runtime.json}
TS_TIMING_JSON=${TS_TIMING_JSON:-${OUTPUT_ROOT}/traceablespeech_runtime.json}
AUDIOSEAL_TIMING_JSON=${AUDIOSEAL_TIMING_JSON:-${OUTPUT_ROOT}/audioseal_runtime.json}
WAVMARK_TIMING_JSON=${WAVMARK_TIMING_JSON:-${OUTPUT_ROOT}/wavmark_runtime.json}

VALLE_PYENV=${VALLE_PYENV:-valle}
EXTERNAL_PYENV=${EXTERNAL_PYENV:-audioseal}
AUTO_SETUP=${AUTO_SETUP:-true}
FORCE=${FORCE:-false}

for switch_name in AUTO_SETUP FORCE; do
    switch_value=${!switch_name}
    if [[ "${switch_value}" != true && "${switch_value}" != false ]]; then
        echo "${switch_name} must be true or false; got ${switch_value}" >&2
        exit 1
    fi
done
if ! [[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_SAMPLES must be a positive integer; got ${N_SAMPLES}" >&2
    exit 1
fi

for path in \
    "${MANIFEST}" \
    "${FIXED_PROMPT_AUDIO}" \
    "${VM_VALLE_CHECKPOINT}" \
    "${TS_VALLE_CHECKPOINT}" \
    "${VOICEMARK_CHECKPOINT}" \
    "${TS_CHECKPOINT}" \
    "${PROJECT_DIR}/bin/infer.py" \
    "${PROJECT_DIR}/watermark_runtime_benchmark.py"; do
    test -f "${path}" || { echo "Missing required file: ${path}" >&2; exit 1; }
done

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_ROOT}" "${VM_OUTPUT_DIR}" "${TS_OUTPUT_DIR}"
cd "${PROJECT_DIR}"

export PYENV_ROOT=${PYENV_ROOT:-${HOME}/.pyenv}
export PATH="${PYENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${AUDIOSEAL_ROOT}/src:${WAVMARK_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/wu25/mrnas04home/.cache/huggingface}

echo "========================================================================"
echo "Watermark runtime benchmark"
echo "Samples             : ${N_SAMPLES}"
echo "Manifest            : ${MANIFEST}"
echo "Fixed prompt        : ${FIXED_PROMPT_AUDIO}"
echo "VoiceMark VALL-E    : ${VM_VALLE_CHECKPOINT}"
echo "VoiceMark watermark : ${VOICEMARK_CHECKPOINT}"
echo "TraceableSpeech     : ${TS_CHECKPOINT}"
echo "Output              : ${OUTPUT_ROOT}"
echo "Force rerun         : ${FORCE}"
echo "========================================================================"

if [[ "${FORCE}" == true || ! -s "${VM_TIMING_JSON}" ]]; then
    echo "[$(date)] Benchmark VALL-E clean + VoiceMark integrated watermark"
    PYENV_VERSION="${VALLE_PYENV}" pyenv exec python3 bin/infer.py \
        --output-dir "${VM_OUTPUT_DIR}" \
        --checkpoint "${VM_VALLE_CHECKPOINT}" \
        --seed-tts-manifest "${MANIFEST}" \
        --seed-tts-num-samples "${N_SAMPLES}" \
        --seed-tts-primary-output watermarked \
        --seed-tts-fixed-prompt-audio "${FIXED_PROMPT_AUDIO}" \
        --seed-tts-fixed-prompt-text "${FIXED_PROMPT_TEXT}" \
        --watermark-backend voicemark \
        --ts-enable false \
        --voicemark-root "${VOICEMARK_ROOT}" \
        --voicemark-config "${VOICEMARK_CONFIG}" \
        --voicemark-st-checkpoint "${VOICEMARK_ST_CHECKPOINT}" \
        --voicemark-checkpoint "${VOICEMARK_CHECKPOINT}" \
        --voicemark-embed-vq1 true \
        --top-k -100 \
        --temperature 1.0 \
        --runtime-timing-json "${VM_TIMING_JSON}" \
        --skip-post-generation-eval
else
    echo "[$(date)] Reuse VoiceMark timing: ${VM_TIMING_JSON}"
fi

if [[ "${FORCE}" == true || ! -s "${TS_TIMING_JSON}" ]]; then
    echo "[$(date)] Benchmark VALL-E clean + TraceableSpeech integrated watermark"
    PYENV_VERSION="${VALLE_PYENV}" pyenv exec python3 bin/infer.py \
        --output-dir "${TS_OUTPUT_DIR}" \
        --checkpoint "${TS_VALLE_CHECKPOINT}" \
        --seed-tts-manifest "${MANIFEST}" \
        --seed-tts-num-samples "${N_SAMPLES}" \
        --seed-tts-primary-output watermarked \
        --seed-tts-fixed-prompt-audio "${FIXED_PROMPT_AUDIO}" \
        --seed-tts-fixed-prompt-text "${FIXED_PROMPT_TEXT}" \
        --watermark-backend traceablespeech \
        --ts-enable true \
        --ts-checkpoint-file "${TS_CHECKPOINT}" \
        --top-k -100 \
        --temperature 1.0 \
        --runtime-timing-json "${TS_TIMING_JSON}" \
        --skip-post-generation-eval
else
    echo "[$(date)] Reuse TraceableSpeech timing: ${TS_TIMING_JSON}"
fi

if ! PYENV_VERSION="${EXTERNAL_PYENV}" pyenv exec python3 -c \
    'import audioseal, julius, soundfile, torch, torchaudio, tqdm, wavmark' \
    >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != true ]]; then
        echo "Missing AudioSeal/WavMark benchmark dependencies." >&2
        exit 1
    fi
    PYENV_VERSION="${EXTERNAL_PYENV}" pyenv exec python3 -m pip install \
        -e "${AUDIOSEAL_ROOT}" -e "${WAVMARK_ROOT}" \
        'julius==0.2.7' 'soundfile==0.12.1' 'tqdm>=4.64'
fi

if [[ "${FORCE}" == true || ! -s "${AUDIOSEAL_TIMING_JSON}" ]]; then
    echo "[$(date)] Benchmark AudioSeal post-hoc embed/extract on VoiceMark clean"
    PYENV_VERSION="${EXTERNAL_PYENV}" pyenv exec python3 \
        watermark_runtime_benchmark.py external \
        --backend audioseal \
        --clean-dir "${VM_OUTPUT_DIR}" \
        --carrier-timing-json "${VM_TIMING_JSON}" \
        --output-json "${AUDIOSEAL_TIMING_JSON}" \
        --recipe-dir "${PROJECT_DIR}" \
        --wavmark-root "${WAVMARK_ROOT}" \
        --device cuda:0
else
    echo "[$(date)] Reuse AudioSeal timing: ${AUDIOSEAL_TIMING_JSON}"
fi

if [[ "${FORCE}" == true || ! -s "${WAVMARK_TIMING_JSON}" ]]; then
    echo "[$(date)] Benchmark WavMark post-hoc embed/extract on VoiceMark clean"
    PYENV_VERSION="${EXTERNAL_PYENV}" pyenv exec python3 \
        watermark_runtime_benchmark.py external \
        --backend wavmark \
        --clean-dir "${VM_OUTPUT_DIR}" \
        --carrier-timing-json "${VM_TIMING_JSON}" \
        --output-json "${WAVMARK_TIMING_JSON}" \
        --recipe-dir "${PROJECT_DIR}" \
        --wavmark-root "${WAVMARK_ROOT}" \
        --device cuda:0
else
    echo "[$(date)] Reuse WavMark timing: ${WAVMARK_TIMING_JSON}"
fi

PYENV_VERSION="${VALLE_PYENV}" pyenv exec python3 \
    watermark_runtime_benchmark.py summarize \
    --voicemark-json "${VM_TIMING_JSON}" \
    --traceablespeech-json "${TS_TIMING_JSON}" \
    --audioseal-json "${AUDIOSEAL_TIMING_JSON}" \
    --wavmark-json "${WAVMARK_TIMING_JSON}" \
    --output-dir "${OUTPUT_ROOT}"

echo "Runtime table: ${OUTPUT_ROOT}/watermark_runtime_summary.txt"
echo "Runtime JSON : ${OUTPUT_ROOT}/watermark_runtime_summary.json"
