#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(cd "${PROJECT_DIR}/../.." && pwd)"
RUN_ID="${PJM_JOBID:-manual}"
OUT_DIR="${SCRIPT_DIR}/exp/test_prepare_speechtokenizer_genkai/${RUN_ID}"
mkdir -p "${OUT_DIR}"

source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh"
conda activate voicemark
export PYTHONPATH="${SCRIPT_DIR}/genkai_compat:${PROJECT_DIR}:${WORKSPACE}/projects/icefall:${WORKSPACE}/projects/NeuMark:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python "${SCRIPT_DIR}/prepare_speechtokenizer_genkai.py" \
    --input-manifest "${SCRIPT_DIR}/data/tokenized_voicemark/cuts_dev.jsonl.gz" \
    --output-manifest "${OUT_DIR}/cuts_dev.jsonl.gz" \
    --output-h5 "${OUT_DIR}/speechtokenizer_dev.h5" \
    --text-tokens "${SCRIPT_DIR}/data/tokenized_voicemark/unique_text_tokens.k2symbols" \
    --st-config "${WORKSPACE}/projects/NeuMark/STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json" \
    --st-checkpoint "${WORKSPACE}/projects/NeuMark/STmodels/pretrained_model/SpeechTokenizer.pt" \
    --preview-dir "${OUT_DIR}/previews" \
    --max-previews 2 \
    --max-cuts 2 \
    --device cuda:0

python - "${OUT_DIR}" <<'PY'
import sys
from pathlib import Path

import h5py
from lhotse import load_manifest_lazy

root = Path(sys.argv[1])
cuts = list(load_manifest_lazy(root / "cuts_dev.jsonl.gz"))
assert len(cuts) == 2, len(cuts)
with h5py.File(root / "speechtokenizer_dev.h5", "r") as store:
    assert store.attrs["encoding_mode"] == "single_utterance_no_padding"
    assert store.attrs["num_quantizers"] == 8
    assert len(store) == 2
    for cut in cuts:
        array = cut.load_features()
        assert array.ndim == 2 and array.shape[1] == 8, array.shape
for index in range(2):
    sample = root / "previews" / f"sample_{index:02d}"
    assert (sample / "00_reference.wav").is_file()
    assert (sample / "01_speechtokenizer_reconstruction.wav").is_file()
print(f"PASS: {len(cuts)} independently encoded cuts; output={root}")
PY
