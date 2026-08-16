#!/usr/bin/env bash
#SBATCH --job-name=vm_fixed_seedtts
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_DIR=/home/wu25/mrnas04home/projects/vall-e/egs/libritts
export EXP_DIR=${VM_EXP_DIR:-exp/valle_voicemark}
export WATERMARK_BACKEND=voicemark
export OUT_DIR=${VM_OUT_DIR:-infer/voicemark_seedtts_fixed_libritts_prompt_epoch40}
export VOICEMARK_CHECKPOINT=${VOICEMARK_CHECKPOINT:-/home/wu25/mrnas04home/projects/VoiceMark/checkpoints/ref_ori.pt}
export SEED_TTS_MANIFEST=${SEED_TTS_MANIFEST:-data/seed_tts_eval/en/meta.lst}
export FIXED_PROMPT_AUDIO=${FIXED_PROMPT_AUDIO:-prompts/8455_210777_000067_000000.wav}

RUN_SEED_TTS_METRICS=${RUN_SEED_TTS_METRICS:-true}
METRICS_RESULT_DIR=${METRICS_RESULT_DIR:-${OUT_DIR}/seed_tts_metrics}

cd "${PROJECT_DIR}"
bash "${PROJECT_DIR}/valid_fixed_prompt_common.sh"

if [[ "${RUN_SEED_TTS_METRICS}" == "true" ]]; then
    echo "[$(date)] Run Seed-TTS WER, Speaker SIM, watermark and quality metrics"
    PROJECT_DIR="${PROJECT_DIR}" \
    AUDIO_DIR="${OUT_DIR}" \
    RESULT_DIR="${METRICS_RESULT_DIR}" \
    MANIFEST="${SEED_TTS_MANIFEST}" \
    PROMPT_WAV="${FIXED_PROMPT_AUDIO}" \
    WATERMARK_BACKEND=voicemark \
    RUN_WER_SIM=true \
    bash "${PROJECT_DIR}/seed_tts_clean_wm_metrics.slurm"
elif [[ "${RUN_SEED_TTS_METRICS}" != "false" ]]; then
    echo "RUN_SEED_TTS_METRICS must be true or false; got ${RUN_SEED_TTS_METRICS}" >&2
    exit 1
fi
