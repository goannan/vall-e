#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(cd "${PROJECT_DIR}/../.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/exp/ab_valle_runtime_precision"
CHECKPOINT="${SCRIPT_DIR}/exp/valle_voicemark/epoch-40.pt"
MANIFEST="${SCRIPT_DIR}/data/tokenized_voicemark/cuts_dev.jsonl.gz"
TEXT_TOKENS="${SCRIPT_DIR}/data/tokenized/unique_text_tokens.k2symbols"
ST_CONFIG="${WORKSPACE}/projects/NeuMark/STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
ST_CHECKPOINT="${WORKSPACE}/projects/NeuMark/STmodels/pretrained_model/SpeechTokenizer.pt"
PROMPT_ID="1462_170142_000021_000003-163"
TARGET_ID="1462_170142_000038_000001-219"

mkdir -p \
    "${OUTPUT_DIR}/legacy_torch113_fp32" \
    "${OUTPUT_DIR}/legacy_source_current_runtime_fp32" \
    "${OUTPUT_DIR}/current_source_torch113_fp32" \
    "${OUTPUT_DIR}/current_source_current_runtime_cpu_fp32" \
    "${OUTPUT_DIR}/current_gpu_cpu_sampling_fp32" \
    "${OUTPUT_DIR}/current_fp32"

source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh"

# Preserved lab-era Python/torch and preserved VALL-E source.  NeuMark is used
# only for its torch-1.13 compatibility import; its ST checkpoint/config hashes
# are identical to VoiceMark's copies.
conda activate valle
cd "${PROJECT_DIR}/src/valle"
export PYTHONPATH="${PROJECT_DIR}/src/valle:${WORKSPACE}/projects/icefall:${WORKSPACE}/projects/NeuMark"
if [ ! -f "${OUTPUT_DIR}/legacy_torch113_fp32/legacy_report.json" ]; then
CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/legacy_runtime_valle_smoke.py" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --text-tokens "${TEXT_TOKENS}" \
    --prompt-cut-id "${PROMPT_ID}" \
    --target-cut-id "${TARGET_ID}" \
    --st-config "${ST_CONFIG}" \
    --st-checkpoint "${ST_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/legacy_torch113_fp32"
fi

# Current VALL-E source under the preserved torch 1.13 runtime (CPU because the
# old CUDA build cannot execute on H100/sm_90).
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${WORKSPACE}/projects/icefall:${WORKSPACE}/projects/NeuMark"
if [ ! -f "${OUTPUT_DIR}/current_source_torch113_fp32/legacy_report.json" ]; then
CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/legacy_runtime_valle_smoke.py" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --text-tokens "${TEXT_TOKENS}" \
    --prompt-cut-id "${PROMPT_ID}" \
    --target-cut-id "${TARGET_ID}" \
    --st-config "${ST_CONFIG}" \
    --st-checkpoint "${ST_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/current_source_torch113_fp32"
fi

# Current source/runtime with autocast disabled.  This differs from the earlier
# diagnosis only in precision, so it isolates BF16 from the other changes.
conda activate voicemark
cd "${PROJECT_DIR}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${PROJECT_DIR}:${WORKSPACE}/projects/icefall:${WORKSPACE}/projects/NeuMark"
if [ ! -f "${OUTPUT_DIR}/current_source_current_runtime_cpu_fp32/legacy_report.json" ]; then
CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/legacy_runtime_valle_smoke.py" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --text-tokens "${TEXT_TOKENS}" \
    --prompt-cut-id "${PROMPT_ID}" \
    --target-cut-id "${TARGET_ID}" \
    --st-config "${ST_CONFIG}" \
    --st-checkpoint "${ST_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/current_source_current_runtime_cpu_fp32"
fi

if [ ! -f "${OUTPUT_DIR}/current_gpu_cpu_sampling_fp32/legacy_report.json" ]; then
python "${SCRIPT_DIR}/legacy_runtime_valle_smoke.py" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --text-tokens "${TEXT_TOKENS}" \
    --prompt-cut-id "${PROMPT_ID}" \
    --target-cut-id "${TARGET_ID}" \
    --st-config "${ST_CONFIG}" \
    --st-checkpoint "${ST_CHECKPOINT}" \
    --sample-on-cpu \
    --output-dir "${OUTPUT_DIR}/current_gpu_cpu_sampling_fp32"
fi

cd "${PROJECT_DIR}/src/valle"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${PROJECT_DIR}/src/valle:${WORKSPACE}/projects/icefall:${WORKSPACE}/projects/NeuMark"
python "${SCRIPT_DIR}/legacy_runtime_valle_smoke.py" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --text-tokens "${TEXT_TOKENS}" \
    --prompt-cut-id "${PROMPT_ID}" \
    --target-cut-id "${TARGET_ID}" \
    --st-config "${ST_CONFIG}" \
    --st-checkpoint "${ST_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/legacy_source_current_runtime_fp32"

cd "${SCRIPT_DIR}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="${PROJECT_DIR}:${WORKSPACE}/projects/icefall:${WORKSPACE}/projects/NeuMark"
if [ ! -f "${OUTPUT_DIR}/current_fp32/diagnosis.json" ]; then
python diagnose_valle_token_vs_checkpoint.py \
    --valle-checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --prompt-cut-id "${PROMPT_ID}" \
    --target-cut-id "${TARGET_ID}" \
    --st-config "${ST_CONFIG}" \
    --st-checkpoint "${ST_CHECKPOINT}" \
    --precision fp32 \
    --output-dir "${OUTPUT_DIR}/current_fp32"
fi
