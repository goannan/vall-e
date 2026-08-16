#!/usr/bin/env bash
#SBATCH --job-name=vm_seed_wer_sim
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_DIR=/home/wu25/mrnas04home/projects/vall-e/egs/libritts
export PROJECT_DIR
export AUDIO_DIR=${AUDIO_DIR:-${PROJECT_DIR}/infer/voicemark_seedtts_en_epoch40}
export MANIFEST=${MANIFEST:-${PROJECT_DIR}/data/seed_tts_eval/en/meta.lst}
export RESULT_DIR=${RESULT_DIR:-${AUDIO_DIR}/seed_tts_metrics}

cd "${PROJECT_DIR}"
exec bash "${PROJECT_DIR}/seed_tts_clean_wm_metrics.slurm"
