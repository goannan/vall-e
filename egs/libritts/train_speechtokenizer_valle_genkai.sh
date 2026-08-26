#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(cd "${PROJECT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

source "${WORKSPACE}/miniconda3/etc/profile.d/conda.sh"
conda activate voicemark

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SCRIPT_DIR}/genkai_compat:${PROJECT_DIR}:${WORKSPACE}/projects/icefall:${WORKSPACE}/projects/NeuMark:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_ASYNC_ERROR_HANDLING=1

# Stage 0: acoustic preparation; 1: AR; 2: NAR; 3: final synthesis check.
START_STAGE="${START_STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-stage)
            START_STAGE="$2"
            shift 2
            ;;
        --stop-stage)
            STOP_STAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-${SCRIPT_DIR}/data/tokenized_voicemark}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/tokenized_speechtokenizer_genkai_v1}"
EXP_ROOT="${EXP_ROOT:-${SCRIPT_DIR}/exp/valle_speechtokenizer_genkai_v1}"
AR_EXP_DIR="${EXP_ROOT}/ar"
NAR_EXP_DIR="${EXP_ROOT}/nar"
PREVIEW_DIR="${EXP_ROOT}/prepared_codec_previews"
FINAL_PREVIEW_DIR="${EXP_ROOT}/final_synthesis_previews"

NEUMARK_ROOT="${NEUMARK_ROOT:-${WORKSPACE}/projects/NeuMark}"
ST_CONFIG="${ST_CONFIG:-${NEUMARK_ROOT}/STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json}"
ST_CHECKPOINT="${ST_CHECKPOINT:-${NEUMARK_ROOT}/STmodels/pretrained_model/SpeechTokenizer.pt}"
SOURCE_TEXT_TOKENS="${SOURCE_TEXT_TOKENS:-${SOURCE_DATA_DIR}/unique_text_tokens.k2symbols}"
TEXT_TOKENS="${DATA_DIR}/unique_text_tokens.k2symbols"

# valle.sh used max-duration=40 with four gradient-accumulation steps.  A
# physical batch of 160 seconds and one accumulation step is 4x larger per
# forward pass while preserving the same effective global batch and LR scale.
WORLD_SIZE="${WORLD_SIZE:-1}"
TRAIN_MAX_DURATION="${TRAIN_MAX_DURATION:-160}"
ACCUMULATE_GRAD_STEPS="${ACCUMULATE_GRAD_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AR_DTYPE="${AR_DTYPE:-bfloat16}"
NAR_DTYPE="${NAR_DTYPE:-float32}"
AR_NUM_EPOCHS="${AR_NUM_EPOCHS:-20}"
NAR_NUM_EPOCHS="${NAR_NUM_EPOCHS:-40}"

mkdir -p "${DATA_DIR}" "${AR_EXP_DIR}" "${NAR_EXP_DIR}" "${EXP_ROOT}/logs"

for required in \
    "${SOURCE_DATA_DIR}/cuts_train.jsonl.gz" \
    "${SOURCE_DATA_DIR}/cuts_dev.jsonl.gz" \
    "${SOURCE_DATA_DIR}/cuts_test.jsonl.gz" \
    "${SOURCE_TEXT_TOKENS}" \
    "${ST_CONFIG}" \
    "${ST_CHECKPOINT}"; do
    if [ ! -f "${required}" ]; then
        echo "Missing required input: ${required}" >&2
        exit 1
    fi
done

if [ -f "${TEXT_TOKENS}" ]; then
    cmp --silent "${SOURCE_TEXT_TOKENS}" "${TEXT_TOKENS}" || {
        echo "Existing ${TEXT_TOKENS} differs from the source vocabulary; refusing to mix data." >&2
        exit 1
    }
else
    cp "${SOURCE_TEXT_TOKENS}" "${TEXT_TOKENS}"
fi

echo "============================================================"
echo " GENKAI SpeechTokenizer VALL-E end-to-end pipeline"
echo " stages             : ${START_STAGE}..${STOP_STAGE}"
echo " source manifests   : ${SOURCE_DATA_DIR}"
echo " new token data     : ${DATA_DIR}"
echo " experiment         : ${EXP_ROOT}"
echo " SpeechTokenizer    : ${ST_CHECKPOINT}"
echo " GPUs/world size    : ${WORLD_SIZE}"
echo " max duration       : ${TRAIN_MAX_DURATION} seconds (4x valle.sh)"
echo " grad accumulation  : ${ACCUMULATE_GRAD_STEPS} (was 4)"
echo "============================================================"
nvidia-smi
python - <<'PY'
import sys, torch, torchaudio
print("python:", sys.version)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
print("visible GPUs:", torch.cuda.device_count())
PY

if [ "${WORLD_SIZE}" -gt "$(python -c 'import torch; print(torch.cuda.device_count())')" ]; then
    echo "WORLD_SIZE=${WORLD_SIZE} exceeds visible GPU count." >&2
    exit 1
fi

if [ "${START_STAGE}" -le 0 ] && [ "${STOP_STAGE}" -ge 0 ]; then
    echo "[$(date)] Stage 0: regenerate all SpeechTokenizer acoustic tokens"
    for split in dev test train; do
        pids=()
        shard_manifests=()
        for ((rank=0; rank<WORLD_SIZE; rank++)); do
            preview_args=()
            if [ "${split}" = "dev" ] && [ "${rank}" -eq 0 ]; then
                preview_args=(--preview-dir "${PREVIEW_DIR}" --max-previews 5)
            fi
            shard_manifest="${DATA_DIR}/cuts_${split}_rank${rank}.jsonl.gz"
            shard_manifests+=("${shard_manifest}")
            CUDA_VISIBLE_DEVICES="${rank}" python prepare_speechtokenizer_genkai.py \
                --input-manifest "${SOURCE_DATA_DIR}/cuts_${split}.jsonl.gz" \
                --output-manifest "${shard_manifest}" \
                --output-h5 "${DATA_DIR}/speechtokenizer_${split}_rank${rank}.h5" \
                --text-tokens "${TEXT_TOKENS}" \
                --st-config "${ST_CONFIG}" \
                --st-checkpoint "${ST_CHECKPOINT}" \
                --rank "${rank}" \
                --world-size "${WORLD_SIZE}" \
                --min-duration 0.5 \
                --max-duration 14 \
                --device cuda:0 \
                "${preview_args[@]}" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "${pid}"
        done
        # This Lhotse version has a single-input bug in `lhotse combine`: it
        # reduces the cuts inside one CutSet instead of returning that CutSet.
        # A single-GPU run does not need a merge, so copy its manifest directly.
        if [ "${WORLD_SIZE}" -eq 1 ]; then
            cp "${shard_manifests[0]}" "${DATA_DIR}/cuts_${split}.jsonl.gz"
        else
            lhotse combine "${shard_manifests[@]}" "${DATA_DIR}/cuts_${split}.jsonl.gz"
        fi
    done
    touch "${DATA_DIR}/.speechtokenizer_prepare.done"
fi

if [ ! -f "${DATA_DIR}/.speechtokenizer_prepare.done" ]; then
    echo "SpeechTokenizer preparation is incomplete: ${DATA_DIR}" >&2
    exit 1
fi

COMMON_TRAIN_ARGS=(
    --world-size "${WORLD_SIZE}"
    --manifest-dir "${DATA_DIR}"
    --text-tokens "${TEXT_TOKENS}"
    --sampling-rate 16000
    --max-duration "${TRAIN_MAX_DURATION}"
    --filter-min-duration 0.5
    --filter-max-duration 14
    --num-workers "${NUM_WORKERS}"
    --num-buckets 6
    --save-every-n 2500
    --valid-interval 5000
    --keep-last-k 20
    --model-name valle
    --share-embedding true
    --norm-first true
    --add-prenet false
    --decoder-dim 1024
    --nhead 16
    --num-decoder-layers 12
    --prefix-mode 1
    --base-lr 0.05
    --warmup-steps 200
    --average-period 0
    --start-batch 0
    --accumulate-grad-steps "${ACCUMULATE_GRAD_STEPS}"
    --oom-check true
)

latest_epoch() {
    local directory="$1"
    local latest=0
    local path number
    shopt -s nullglob
    for path in "${directory}"/epoch-*.pt; do
        number="${path##*/epoch-}"
        number="${number%.pt}"
        if [[ "${number}" =~ ^[0-9]+$ ]] && [ "${number}" -gt "${latest}" ]; then
            latest="${number}"
        fi
    done
    shopt -u nullglob
    echo "${latest}"
}

if [ "${START_STAGE}" -le 1 ] && [ "${STOP_STAGE}" -ge 1 ]; then
    echo "[$(date)] Stage 1: train AR VALL-E"
    ar_latest="$(latest_epoch "${AR_EXP_DIR}")"
    if [ "${ar_latest}" -ge "${AR_NUM_EPOCHS}" ] && [ -f "${AR_EXP_DIR}/best-valid-loss.pt" ]; then
        echo "AR stage already complete at epoch ${ar_latest}; skipping."
    else
        ar_start=$((ar_latest + 1))
        python bin/trainer.py "${COMMON_TRAIN_ARGS[@]}" \
            --train-stage 1 \
            --dtype "${AR_DTYPE}" \
            --num-epochs "${AR_NUM_EPOCHS}" \
            --start-epoch "${ar_start}" \
            --exp-dir "${AR_EXP_DIR}"
    fi
    test -f "${AR_EXP_DIR}/best-valid-loss.pt"
    touch "${AR_EXP_DIR}/.ar.done"
fi

if [ ! -f "${AR_EXP_DIR}/.ar.done" ]; then
    echo "AR stage is incomplete: ${AR_EXP_DIR}" >&2
    exit 1
fi

if [ "${START_STAGE}" -le 2 ] && [ "${STOP_STAGE}" -ge 2 ]; then
    echo "[$(date)] Stage 2: initialize from AR and train NAR VALL-E"
    if [ ! -f "${NAR_EXP_DIR}/.initialized_from_ar" ]; then
        cp "${AR_EXP_DIR}/best-valid-loss.pt" "${NAR_EXP_DIR}/epoch-2.pt"
        touch "${NAR_EXP_DIR}/.initialized_from_ar"
    fi
    nar_latest="$(latest_epoch "${NAR_EXP_DIR}")"
    if [ "${nar_latest}" -ge "${NAR_NUM_EPOCHS}" ] && [ -f "${NAR_EXP_DIR}/best-valid-loss.pt" ]; then
        echo "NAR stage already complete at epoch ${nar_latest}; skipping."
    else
        if [ "${nar_latest}" -le 2 ]; then
            nar_start=3
        else
            nar_start=$((nar_latest + 1))
        fi
        python bin/trainer.py "${COMMON_TRAIN_ARGS[@]}" \
            --train-stage 2 \
            --dtype "${NAR_DTYPE}" \
            --num-epochs "${NAR_NUM_EPOCHS}" \
            --start-epoch "${nar_start}" \
            --exp-dir "${NAR_EXP_DIR}"
    fi
    test -f "${NAR_EXP_DIR}/best-valid-loss.pt"
    touch "${NAR_EXP_DIR}/.nar.done"
fi

if [ ! -f "${NAR_EXP_DIR}/.nar.done" ]; then
    echo "NAR stage is incomplete: ${NAR_EXP_DIR}" >&2
    exit 1
fi

ln -sfn "nar/best-valid-loss.pt" "${EXP_ROOT}/final.pt"

if [ "${START_STAGE}" -le 3 ] && [ "${STOP_STAGE}" -ge 3 ]; then
    echo "[$(date)] Stage 3: synthesize five final validation previews"
    python test_valle_token_synthesis.py \
        --valle-checkpoint "${EXP_ROOT}/final.pt" \
        --manifest "${DATA_DIR}/cuts_dev.jsonl.gz" \
        --text-tokens "${TEXT_TOKENS}" \
        --st-config "${ST_CONFIG}" \
        --st-checkpoint "${ST_CHECKPOINT}" \
        --num-samples 5 \
        --min-duration 3 \
        --max-duration 10 \
        --precision fp32 \
        --sample-on-cpu \
        --output-dir "${FINAL_PREVIEW_DIR}"
fi

echo "[$(date)] Pipeline complete"
echo "Final checkpoint: ${EXP_ROOT}/final.pt"
echo "Validation WAVs: ${FINAL_PREVIEW_DIR}"
