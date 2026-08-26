#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate voicemark || true
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}/../../:${SCRIPT_DIR}/../../../icefall:${SCRIPT_DIR}/../../../NeuMark:${PYTHONPATH:-}"

RUN_ID="${PJM_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/exp/test_generate_valle_native_tokens/${RUN_ID}}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
MIN_DURATION="${MIN_DURATION:-3.0}"
MAX_DURATION="${MAX_DURATION:-10.0}"
mkdir -p "$OUT_DIR"

python3 generate_valle_native_dataset.py \
    --valle-checkpoint exp/valle_voicemark/epoch-40.pt \
    --input-manifest data/tokenized_voicemark/cuts_dev.jsonl.gz \
    --output-manifest "${OUT_DIR}/cuts_dev_valle_native.jsonl.gz" \
    --output-h5 "${OUT_DIR}/valle_native_dev.h5" \
    --min-duration "$MIN_DURATION" \
    --max-duration "$MAX_DURATION" \
    --max-samples "$NUM_SAMPLES" \
    --max-generation-attempts 3 \
    --preview-dir "${OUT_DIR}/previews" \
    --max-previews "$NUM_SAMPLES" \
    --device cuda:0

python3 - "$OUT_DIR" <<'PY'
import gzip
import json
import sys
from pathlib import Path

import h5py

out_dir = Path(sys.argv[1])
h5_path = out_dir / "valle_native_dev.h5"
manifest_path = out_dir / "cuts_dev_valle_native.jsonl.gz"

with h5py.File(h5_path, "r") as store:
    assert store.attrs["generation_version"] == 4
    assert store.attrs["prompt_token_source"] == "single_wav_speechtokenizer_encode"
    keys = list(store.keys())
    assert keys, "The smoke test generated no accepted token sequence"
    for key in keys:
        dataset = store[key]
        assert dataset.shape[1] == 8
        assert dataset.attrs["prompt_token_source"] == "single_wav_speechtokenizer_encode"
        assert Path(dataset.attrs["prompt_audio_path"]).is_file()

with gzip.open(manifest_path, "rt") as stream:
    cuts = [json.loads(line) for line in stream if line.strip()]
assert cuts, "The output manifest is empty"
for cut in cuts:
    custom = cut["custom"]
    assert custom["generation_version"] == 4
    assert custom["prompt_token_source"] == "single_wav_speechtokenizer_encode"
    assert Path(custom["prompt_audio_path"]).is_file()
    assert 0.25 <= custom["generated_to_target_duration_ratio"] <= 3.0

preview_dirs = sorted((out_dir / "previews").glob("sample_*"))
assert len(preview_dirs) == len(cuts)
for sample_dir in preview_dirs:
    assert (sample_dir / "00_prompt.wav").is_file()
    assert (sample_dir / "01_target_reference.wav").is_file()
    assert (sample_dir / "02_valle_generated.wav").is_file()
    assert (sample_dir / "metadata.json").is_file()

print(f"PASS: {len(cuts)} generated cuts and preview triplets; output={out_dir}")
PY
