#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 14-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

# Runtime environment. Set PYENV_VERSION=none if your sbatch environment is
# already activated.
PYENV_VERSION=${PYENV_VERSION:-valle}
if [ "${PYENV_VERSION}" != "none" ]; then
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  if [ -d "${PYENV_ROOT}/bin" ]; then
    export PATH="${PYENV_ROOT}/bin:${PATH}"
  fi
  if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    pyenv activate "${PYENV_VERSION}"
  fi
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Data/tokenization settings.
RUN_PREPARE=${RUN_PREPARE:-true}
PREPARE_STAGE=${PREPARE_STAGE:-0}
PREPARE_STOP_STAGE=${PREPARE_STOP_STAGE:-3}
DATASET_PARTS=${DATASET_PARTS:---dataset-parts all}
N_JOBS=${N_JOBS:-16}

export AUDIO_EXTRACTOR=${AUDIO_EXTRACTOR:-VoiceMark}
export AUDIO_FEATS_DIR=${AUDIO_FEATS_DIR:-data/tokenized_voicemark}
export VOICEMARK_ROOT=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}
export VOICEMARK_CONFIG=${VOICEMARK_CONFIG:-STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json}
export VOICEMARK_ST_CHECKPOINT=${VOICEMARK_ST_CHECKPOINT:-STmodels/pretrained_model/SpeechTokenizer.pt}
export VOICEMARK_CHECKPOINT=${VOICEMARK_CHECKPOINT:-train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt}
export VOICEMARK_EMBED_VQ1=${VOICEMARK_EMBED_VQ1:-true}

# Training settings.
EXP_DIR=${EXP_DIR:-exp/valle_voicemark}
SAMPLING_RATE=${SAMPLING_RATE:-16000}
RUN_AR=${RUN_AR:-true}
RUN_NAR=${RUN_NAR:-true}

AR_MAX_DURATION=${AR_MAX_DURATION:-80}
AR_DTYPE=${AR_DTYPE:-float16}
AR_NUM_EPOCHS=${AR_NUM_EPOCHS:-20}
AR_ACCUMULATE_GRAD_STEPS=${AR_ACCUMULATE_GRAD_STEPS:-4}

NAR_MAX_DURATION=${NAR_MAX_DURATION:-40}
NAR_DTYPE=${NAR_DTYPE:-float32}
NAR_NUM_EPOCHS=${NAR_NUM_EPOCHS:-40}
NAR_START_EPOCH=${NAR_START_EPOCH:-3}
NAR_ACCUMULATE_GRAD_STEPS=${NAR_ACCUMULATE_GRAD_STEPS:-4}

COMMON_TRAIN_ARGS=(
  --filter-min-duration 0.5
  --filter-max-duration 14
  --manifest-dir "${AUDIO_FEATS_DIR}"
  --sampling-rate "${SAMPLING_RATE}"
  --num-buckets 6
  --save-every-n 10000
  --valid-interval 20000
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
  --exp-dir "${EXP_DIR}"
)

echo "[$(date)] VoiceMark VALL-E pipeline start"
echo "AUDIO_FEATS_DIR=${AUDIO_FEATS_DIR}"
echo "EXP_DIR=${EXP_DIR}"
echo "DATASET_PARTS=${DATASET_PARTS}"

if [ "${RUN_PREPARE}" = "true" ]; then
  echo "[$(date)] Stage A: prepare/tokenize with ${AUDIO_EXTRACTOR}"
  bash prepare.sh \
    --stage "${PREPARE_STAGE}" \
    --stop-stage "${PREPARE_STOP_STAGE}" \
    --dataset-parts "${DATASET_PARTS}" \
    --nj "${N_JOBS}" \
    --audio-extractor "${AUDIO_EXTRACTOR}" \
    --audio-feats-dir "${AUDIO_FEATS_DIR}" \
    --voicemark-root "${VOICEMARK_ROOT}" \
    --voicemark-config "${VOICEMARK_CONFIG}" \
    --voicemark-st-checkpoint "${VOICEMARK_ST_CHECKPOINT}" \
    --voicemark-checkpoint "${VOICEMARK_CHECKPOINT}" \
    --voicemark-embed-vq1 "${VOICEMARK_EMBED_VQ1}"
else
  echo "[$(date)] Stage A skipped: RUN_PREPARE=false"
fi

if [ "${RUN_AR}" = "true" ]; then
  echo "[$(date)] Stage B: AR training"
  python3 bin/trainer.py "${COMMON_TRAIN_ARGS[@]}" \
    --train-stage 1 \
    --max-duration "${AR_MAX_DURATION}" \
    --dtype "${AR_DTYPE}" \
    --num-epochs "${AR_NUM_EPOCHS}" \
    --start-epoch 1 \
    --accumulate-grad-steps "${AR_ACCUMULATE_GRAD_STEPS}"
fi

if [ "${RUN_NAR}" = "true" ]; then
  echo "[$(date)] Stage C: NAR training"
  if [ ! -f "${EXP_DIR}/best-valid-loss.pt" ]; then
    echo "Missing ${EXP_DIR}/best-valid-loss.pt; cannot start NAR training." >&2
    exit 1
  fi
  cp "${EXP_DIR}/best-valid-loss.pt" "${EXP_DIR}/epoch-2.pt"
  python3 bin/trainer.py "${COMMON_TRAIN_ARGS[@]}" \
    --train-stage 2 \
    --max-duration "${NAR_MAX_DURATION}" \
    --dtype "${NAR_DTYPE}" \
    --num-epochs "${NAR_NUM_EPOCHS}" \
    --start-epoch "${NAR_START_EPOCH}" \
    --accumulate-grad-steps "${NAR_ACCUMULATE_GRAD_STEPS}"
fi

echo "[$(date)] VoiceMark VALL-E pipeline done"
