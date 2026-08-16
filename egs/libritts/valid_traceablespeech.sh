#!/usr/bin/env bash
#SBATCH --job-name=ts_fixed_seedtts
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_DIR=/home/wu25/mrnas04home/projects/vall-e/egs/libritts
export EXP_DIR=${TS_EXP_DIR:-exp/valle}
export WATERMARK_BACKEND=traceablespeech
export OUT_DIR=${TS_OUT_DIR:-infer/traceablespeech_seedtts_fixed_libritts_prompt_epoch40}

cd "${PROJECT_DIR}"
exec bash "${PROJECT_DIR}/valid_fixed_prompt_common.sh"
