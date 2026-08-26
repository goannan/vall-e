#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs exp/test_valle_synthesis

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate voicemark || true
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${SCRIPT_DIR}/../../:${SCRIPT_DIR}/../../../icefall:${SCRIPT_DIR}/../../../NeuMark:${PYTHONPATH:-}"

python3 verify_valle_synthesis.py "$@"
