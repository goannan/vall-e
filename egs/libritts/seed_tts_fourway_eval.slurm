#!/usr/bin/env bash
#SBATCH --job-name=seed_fourway_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/wu25/mrnas04home/projects/vall-e/egs/libritts}
cd "${PROJECT_DIR}"

MANIFEST=${MANIFEST:-${PROJECT_DIR}/data/seed_tts_eval/en/meta.lst}
FIXED_PROMPT_WAV=${FIXED_PROMPT_WAV:-${PROJECT_DIR}/prompts/8455_210777_000067_000000.wav}

TS_VALLE_PT=${TS_VALLE_PT:-exp/valle/epoch-40.pt}
VM_VALLE_PT=${VM_VALLE_PT:-exp/valle_voicemark/epoch-40.pt}

TS_SEED_DIR=${TS_SEED_DIR:-${PROJECT_DIR}/infer/traceablespeech_seedtts_en_epoch40}
TS_FIXED_DIR=${TS_FIXED_DIR:-${PROJECT_DIR}/infer/traceablespeech_seedtts_fixed_libritts_prompt_epoch40}
VM_SEED_DIR=${VM_SEED_DIR:-${PROJECT_DIR}/infer/voicemark_seedtts_en_epoch40}
VM_FIXED_DIR=${VM_FIXED_DIR:-${PROJECT_DIR}/infer/voicemark_seedtts_fixed_libritts_prompt_epoch40}

TS_SEED_RESULT=${TS_SEED_RESULT:-${TS_SEED_DIR}/seed_tts_metrics}
TS_FIXED_RESULT=${TS_FIXED_RESULT:-${TS_FIXED_DIR}/seed_tts_metrics}
VM_SEED_RESULT=${VM_SEED_RESULT:-${VM_SEED_DIR}/seed_tts_metrics}
VM_FIXED_RESULT=${VM_FIXED_RESULT:-${VM_FIXED_DIR}/seed_tts_metrics}
FOURWAY_RESULT_DIR=${FOURWAY_RESULT_DIR:-${PROJECT_DIR}/infer/seed_tts_fourway_metrics}

# These are forwarded to the per-directory evaluator.
RUN_WATERMARK_QUALITY=${RUN_WATERMARK_QUALITY:-true}
RUN_WER_SIM=${RUN_WER_SIM:-true}
RUN_UTMOS=${RUN_UTMOS:-true}
WATERMARK_MODE=${WATERMARK_MODE:-auto}
SKIP_CODEC_ATTACKS=${SKIP_CODEC_ATTACKS:-false}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-false}
AUTO_SETUP=${AUTO_SETUP:-true}
STRICT_ALL_PAIRS=${STRICT_ALL_PAIRS:-true}
WHISPER_BATCH_SIZE=${WHISPER_BATCH_SIZE:-4}

test -f "${PROJECT_DIR}/${TS_VALLE_PT}" || {
    echo "Missing TraceableSpeech VALL-E checkpoint: ${PROJECT_DIR}/${TS_VALLE_PT}" >&2
    exit 1
}
test -f "${PROJECT_DIR}/${VM_VALLE_PT}" || {
    echo "Missing VoiceMark VALL-E checkpoint: ${PROJECT_DIR}/${VM_VALLE_PT}" >&2
    exit 1
}
test -f "${MANIFEST}" || { echo "Missing manifest: ${MANIFEST}" >&2; exit 1; }
test -f "${FIXED_PROMPT_WAV}" || {
    echo "Missing fixed LibriTTS prompt: ${FIXED_PROMPT_WAV}" >&2
    exit 1
}
for audio_dir in "${TS_SEED_DIR}" "${TS_FIXED_DIR}" "${VM_SEED_DIR}" "${VM_FIXED_DIR}"; do
    test -d "${audio_dir}" || { echo "Missing audio directory: ${audio_dir}" >&2; exit 1; }
done

mkdir -p logs "${FOURWAY_RESULT_DIR}"

run_case() {
    local case_name=$1
    local audio_dir=$2
    local result_dir=$3
    local prompt_wav=$4

    echo
    echo "========================================================================"
    echo "Evaluate ${case_name}"
    echo "Audio  : ${audio_dir}"
    echo "Prompt : ${prompt_wav:-per-row Seed-TTS prompts}"
    echo "Result : ${result_dir}"
    echo "========================================================================"

    PROJECT_DIR="${PROJECT_DIR}" \
    AUDIO_DIR="${audio_dir}" \
    RESULT_DIR="${result_dir}" \
    MANIFEST="${MANIFEST}" \
    PROMPT_WAV="${prompt_wav}" \
    RUN_WATERMARK_QUALITY="${RUN_WATERMARK_QUALITY}" \
    RUN_WER_SIM="${RUN_WER_SIM}" \
    RUN_UTMOS="${RUN_UTMOS}" \
    WATERMARK_MODE="${WATERMARK_MODE}" \
    WATERMARK_BACKEND=auto \
    SKIP_CODEC_ATTACKS="${SKIP_CODEC_ATTACKS}" \
    FORCE_RECOMPUTE="${FORCE_RECOMPUTE}" \
    AUTO_SETUP="${AUTO_SETUP}" \
    STRICT_ALL_PAIRS="${STRICT_ALL_PAIRS}" \
    WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE}" \
    bash "${PROJECT_DIR}/seed_tts_clean_wm_metrics.slurm"
}

# Per-row Seed-TTS prompt cases.
run_case "TraceableSpeech / Seed prompt" "${TS_SEED_DIR}" "${TS_SEED_RESULT}" ""
run_case "VoiceMark / Seed prompt" "${VM_SEED_DIR}" "${VM_SEED_RESULT}" ""

# Fixed LibriTTS prompt cases.  PROMPT_WAV must be forwarded so Prompt SIM is
# not accidentally measured against each manifest row's original prompt.
run_case "TraceableSpeech / fixed LibriTTS prompt" "${TS_FIXED_DIR}" "${TS_FIXED_RESULT}" "${FIXED_PROMPT_WAV}"
run_case "VoiceMark / fixed LibriTTS prompt" "${VM_FIXED_DIR}" "${VM_FIXED_RESULT}" "${FIXED_PROMPT_WAV}"

python3 "${PROJECT_DIR}/seed_tts_fourway_summary.py" \
    --recipe-dir "${PROJECT_DIR}" \
    --output-dir "${FOURWAY_RESULT_DIR}" \
    --case ts_seed traceablespeech manifest "${TS_VALLE_PT}" \
        "${TS_SEED_DIR}" "${TS_SEED_RESULT}/all_metrics_summary.json" \
    --case ts_fixed traceablespeech fixed "${TS_VALLE_PT}" \
        "${TS_FIXED_DIR}" "${TS_FIXED_RESULT}/all_metrics_summary.json" \
    --case vm_seed voicemark manifest "${VM_VALLE_PT}" \
        "${VM_SEED_DIR}" "${VM_SEED_RESULT}/all_metrics_summary.json" \
    --case vm_fixed voicemark fixed "${VM_VALLE_PT}" \
        "${VM_FIXED_DIR}" "${VM_FIXED_RESULT}/all_metrics_summary.json"

echo "Four-way table: ${FOURWAY_RESULT_DIR}/fourway_metrics_summary.txt"
echo "Four-way JSON : ${FOURWAY_RESULT_DIR}/fourway_metrics_summary.json"
