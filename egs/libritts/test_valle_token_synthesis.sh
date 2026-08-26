#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs exp/test_valle_tokens_short

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate voicemark || true
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}/../../:${SCRIPT_DIR}/../../../icefall:${SCRIPT_DIR}/../../../NeuMark:${PYTHONPATH:-}"

python3 test_valle_token_synthesis.py "$@"
