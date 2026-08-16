#!/usr/bin/env bash
#SBATCH --job-name=fixed_seedtts
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_DIR=/home/wu25/mrnas04home/projects/vall-e/egs/libritts
backend=${VALID_BACKEND:-traceablespeech}

case "${backend}" in
    voicemark)
        entry=${PROJECT_DIR}/valid_voicemark.sh
        ;;
    traceablespeech)
        entry=${PROJECT_DIR}/valid_traceablespeech.sh
        ;;
    *)
        echo "VALID_BACKEND must be voicemark or traceablespeech; got: ${backend}" >&2
        exit 1
        ;;
esac

echo "Dispatch fixed-prompt validation to: ${backend}"
exec bash "${entry}"
