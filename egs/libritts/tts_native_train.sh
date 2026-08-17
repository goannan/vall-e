#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Conda environment activation
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate voicemark || true
fi

echo "=========================================================="
echo " Starting NeuMark Training                                "
echo " Host:        $(hostname)                                 "
echo " Time:        $(date)                                     "
echo " Python:      $(which python3)                            "
echo " CUDA Device: ${CUDA_VISIBLE_DEVICES:-auto}               "
echo "=========================================================="

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

CONFIG_PATH="${1:-${SCRIPT_DIR}/config_tts_native.json}"

python3 tts_native_train.py --config "${CONFIG_PATH}"
