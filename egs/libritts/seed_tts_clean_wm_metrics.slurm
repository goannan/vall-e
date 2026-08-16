#!/usr/bin/env bash
#SBATCH --job-name=seed_all_metrics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

# Examples:
#   Standard per-row Seed-TTS prompts:
#     sbatch --export=ALL,AUDIO_DIR=/path/to/infer_dir seed_tts_clean_wm_metrics.slurm
#   Fixed-prompt synthesis (required for a correct Prompt SIM):
#     sbatch --export=ALL,AUDIO_DIR=/path/to/infer_dir,PROMPT_WAV=/path/to/prompt.wav seed_tts_clean_wm_metrics.slurm
#   Force detector + every DSP/codec attack to run again:
#     sbatch --export=ALL,AUDIO_DIR=/path/to/infer_dir,WATERMARK_MODE=recompute seed_tts_clean_wm_metrics.slurm
#   Run only WER/SIM:
#     sbatch --export=ALL,AUDIO_DIR=/path/to/infer_dir,RUN_WATERMARK_QUALITY=false,RUN_UTMOS=false seed_tts_clean_wm_metrics.slurm
#   Run only watermark robustness and PESQ/STOI/SI-SNR:
#     sbatch --export=ALL,AUDIO_DIR=/path/to/infer_dir,RUN_WER_SIM=false,RUN_UTMOS=false seed_tts_clean_wm_metrics.slurm
#   Run only UTMOS (no other metric modules):
#     sbatch --export=ALL,AUDIO_DIR=/path/to/infer_dir,RUN_UTMOS=true,RUN_WATERMARK_QUALITY=false,RUN_WER_SIM=false seed_tts_clean_wm_metrics.slurm

# sbatch copies this file into /var/lib/slurm/slurmd/job*/ before execution, so
# BASH_SOURCE points at the Slurm spool instead of this recipe directory.
# SLURM_SUBMIT_DIR is the directory from which the user ran sbatch.
DEFAULT_PROJECT_DIR=/home/wu25/mrnas04home/projects/vall-e/egs/libritts
if [[ -n "${PROJECT_DIR:-}" ]]; then
    SCRIPT_DIR=$(cd "${PROJECT_DIR}" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/seed_tts_metric_utils.py" ]]; then
    SCRIPT_DIR=$(cd "${SLURM_SUBMIT_DIR}" && pwd)
else
    SCRIPT_DIR=${DEFAULT_PROJECT_DIR}
fi
test -f "${SCRIPT_DIR}/seed_tts_metric_utils.py" || {
    echo "Cannot locate the libritts recipe directory: ${SCRIPT_DIR}" >&2
    echo "Set PROJECT_DIR=/path/to/vall-e/egs/libritts when submitting." >&2
    exit 1
}
cd "${SCRIPT_DIR}"

AUDIO_DIR=${AUDIO_DIR:-${SCRIPT_DIR}/infer/voicemark_seedtts_en_epoch40}
MANIFEST=${MANIFEST:-${SCRIPT_DIR}/data/seed_tts_eval/en/meta.lst}
RESULT_DIR=${RESULT_DIR:-${AUDIO_DIR}/seed_tts_metrics}
# Leave empty for the standard Seed-TTS protocol (each row uses its manifest
# prompt).  Fixed-prompt synthesis must set this to the actual fixed prompt.
PROMPT_WAV=${PROMPT_WAV:-}
EVAL_REPO=${EVAL_REPO:-${SCRIPT_DIR}/tools/seed-tts-eval}
EVAL_COMMIT=${EVAL_COMMIT:-752f4297f090c46bb1a55a1f7439e5944ddefe8d}
WAVLM_CHECKPOINT=${WAVLM_CHECKPOINT:-${SCRIPT_DIR}/models/wavlm_large_finetune.pth}
WAVLM_GDRIVE_ID=${WAVLM_GDRIVE_ID:-1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP}
# Use the same pyenv as the VALL-E pipeline. Keep this as a separate variable
# so an inherited PYENV_VERSION cannot silently select another environment.
EVAL_PYENV=valle
WHISPER_BATCH_SIZE=${WHISPER_BATCH_SIZE:-4}
AUTO_SETUP=${AUTO_SETUP:-true}
QUALITY_SR=${QUALITY_SR:-16000}
STRICT_ALL_PAIRS=${STRICT_ALL_PAIRS:-true}
FORCE_RECOMPUTE=${FORCE_RECOMPUTE:-false}
# Three independently switchable evaluation modules.
RUN_WATERMARK_QUALITY=${RUN_WATERMARK_QUALITY:-true}
RUN_WER_SIM=${RUN_WER_SIM:-true}
RUN_UTMOS=${RUN_UTMOS:-true}
UTMOS_REPO=${UTMOS_REPO:-tarepan/SpeechMOS:v1.2.0}
UTMOS_MODEL=${UTMOS_MODEL:-utmos22_strong}
UTMOS_DEVICE=${UTMOS_DEVICE:-cuda:0}

# auto: reuse a complete generation-time watermark table when it covers every
# pair; otherwise run the detector/attack suite on the existing WAVs.
# recompute: always rerun all watermark attacks. reuse: require existing files.
WATERMARK_MODE=${WATERMARK_MODE:-auto}
WATERMARK_BACKEND=${WATERMARK_BACKEND:-auto}
SKIP_CODEC_ATTACKS=${SKIP_CODEC_ATTACKS:-false}
TS_CHECKPOINT=${TS_CHECKPOINT:-/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000}
TS_CONFIG=${TS_CONFIG:-/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json}
VOICEMARK_ROOT=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}
VOICEMARK_CONFIG=${VOICEMARK_CONFIG:-STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json}
VOICEMARK_ST_CHECKPOINT=${VOICEMARK_ST_CHECKPOINT:-STmodels/pretrained_model/SpeechTokenizer.pt}
VOICEMARK_CHECKPOINT=${VOICEMARK_CHECKPOINT:-train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt}

mkdir -p logs "${RESULT_DIR}" "$(dirname "${WAVLM_CHECKPOINT}")" "$(dirname "${EVAL_REPO}")"

for switch_name in RUN_WATERMARK_QUALITY RUN_WER_SIM RUN_UTMOS; do
    switch_value=${!switch_name}
    if [[ "${switch_value}" != "true" && "${switch_value}" != "false" ]]; then
        echo "${switch_name} must be true or false; got ${switch_value}" >&2
        exit 1
    fi
done
if [[ "${RUN_WATERMARK_QUALITY}" == "false" && \
      "${RUN_WER_SIM}" == "false" && "${RUN_UTMOS}" == "false" ]]; then
    echo "All evaluation modules are disabled; nothing to do." >&2
    exit 1
fi

export PYENV_ROOT=${PYENV_ROOT:-${HOME}/.pyenv}
export PATH="${PYENV_ROOT}/bin:${PATH}"
export PYENV_VERSION="${EVAL_PYENV}"
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    pyenv shell "${EVAL_PYENV}"
fi

echo "Audio directory : ${AUDIO_DIR}"
echo "Manifest        : ${MANIFEST}"
echo "Prompt WAV      : ${PROMPT_WAV:-per-row Seed-TTS manifest prompt}"
echo "Results         : ${RESULT_DIR}"
echo "Watermark mode  : ${WATERMARK_MODE} (${WATERMARK_BACKEND})"
echo "Run WM+quality  : ${RUN_WATERMARK_QUALITY}"
echo "Run WER+SIM     : ${RUN_WER_SIM}"
echo "Run UTMOS       : ${RUN_UTMOS} (${UTMOS_REPO} / ${UTMOS_MODEL})"
echo "Evaluation env  : ${EVAL_PYENV}"
echo "Python          : $(command -v python3)"
echo "CUDA devices    : ${CUDA_VISIBLE_DEVICES:-assigned by Slurm}"

test -d "${AUDIO_DIR}" || { echo "Missing AUDIO_DIR: ${AUDIO_DIR}" >&2; exit 1; }
if [[ "${RUN_WER_SIM}" == "true" ]]; then
    test -f "${MANIFEST}" || { echo "Missing MANIFEST: ${MANIFEST}" >&2; exit 1; }
fi

if [[ "${RUN_WER_SIM}" == "true" && ! -d "${EVAL_REPO}/.git" ]]; then
    if [[ "${AUTO_SETUP}" != "true" ]]; then
        echo "Missing official Seed-TTS evaluator: ${EVAL_REPO}" >&2
        exit 1
    fi
    git clone https://github.com/BytedanceSpeech/seed-tts-eval.git "${EVAL_REPO}"
fi
if [[ "${RUN_WER_SIM}" == "true" ]]; then
    git -C "${EVAL_REPO}" checkout --detach "${EVAL_COMMIT}"
fi

# Install metric-only dependencies without replacing VALL-E's working
# torch/torchaudio stack. Transformers 4.36.2 supports Whisper-large-v3 and
# remains compatible with the torch 1.13 build in this environment.
if [[ "${RUN_WER_SIM}" == "true" ]] && \
   ! python3 -c 'import fire, s3prl, zhon' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != "true" ]]; then
        echo "Missing metric dependencies. Run with AUTO_SETUP=true." >&2
        exit 1
    fi
    python3 -m pip install --no-deps 'fire==0.7.1' 's3prl==0.4.18' 'zhon==2.1.1'
fi
if [[ "${RUN_WER_SIM}" == "true" ]] && \
   ! python3 -c 'import transformers' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != "true" ]]; then
        echo "Missing transformers in ${EVAL_PYENV}." >&2
        exit 1
    fi
    python3 -m pip install 'transformers==4.36.2'
fi
if [[ "${RUN_WER_SIM}" == "true" ]] && \
   ! python3 -c 'import gdown' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != "true" ]]; then
        echo "Missing gdown in ${EVAL_PYENV}." >&2
        exit 1
    fi
    python3 -m pip install 'gdown==4.7.3'
fi
if [[ "${RUN_WATERMARK_QUALITY}" == "true" ]] && \
   ! python3 -c 'import librosa, pesq, pystoi, soundfile' >/dev/null 2>&1; then
    if [[ "${AUTO_SETUP}" != "true" ]]; then
        echo "Missing PESQ/STOI audio-quality dependencies in ${EVAL_PYENV}." >&2
        exit 1
    fi
    python3 -m pip install \
        'librosa==0.10.2.post1' 'pesq==0.0.4' \
        'pystoi==0.4.1' 'soundfile==0.12.1'
fi
python3 -c 'import torch, torchaudio; print("torch=", torch.__version__, "torchaudio=", torchaudio.__version__)'

if [[ "${RUN_WER_SIM}" == "true" && ! -s "${WAVLM_CHECKPOINT}" ]]; then
    if [[ "${AUTO_SETUP}" != "true" ]]; then
        echo "Missing WavLM checkpoint: ${WAVLM_CHECKPOINT}" >&2
        exit 1
    fi
    WAVLM_GDRIVE_ID="${WAVLM_GDRIVE_ID}" \
    WAVLM_CHECKPOINT="${WAVLM_CHECKPOINT}" \
    python3 - <<'PY'
import os

import gdown

output = gdown.download(
    id=os.environ["WAVLM_GDRIVE_ID"],
    output=os.environ["WAVLM_CHECKPOINT"],
    quiet=False,
)
if output is None:
    raise RuntimeError("gdown failed to download the WavLM checkpoint")
PY
fi

if [[ "${RUN_WER_SIM}" == "true" ]]; then
prepare_cmd=(
    python3 seed_tts_metric_utils.py prepare
    --manifest "${MANIFEST}"
    --audio-dir "${AUDIO_DIR}"
    --output-dir "${RESULT_DIR}"
)
if [[ -n "${PROMPT_WAV}" ]]; then
    prepare_cmd+=(--prompt-wav "${PROMPT_WAV}")
fi
"${prepare_cmd[@]}"

# Pair lists are regenerated on every run.  If AUDIO_DIR, manifest targets, or
# PROMPT_WAV changed, cached WER/SIM outputs no longer describe these pairs.
PAIR_HASH=$(sha256sum \
    "${RESULT_DIR}/clean_pairs.lst" \
    "${RESULT_DIR}/wm_pairs.lst" \
    "${RESULT_DIR}/clean_wm_pairs.lst" | sha256sum | awk '{print $1}')
PAIR_HASH_FILE=${RESULT_DIR}/pairs.sha256
if [[ -s "${PAIR_HASH_FILE}" && "$(<"${PAIR_HASH_FILE}")" != "${PAIR_HASH}" ]]; then
    echo "[$(date)] Pair definition changed; invalidate cached WER/SIM"
    rm -f \
        "${RESULT_DIR}/clean/wer_summary.json" \
        "${RESULT_DIR}/clean/wer_details.jsonl" \
        "${RESULT_DIR}/clean/sim_raw.tsv" \
        "${RESULT_DIR}/wm/wer_summary.json" \
        "${RESULT_DIR}/wm/wer_details.jsonl" \
        "${RESULT_DIR}/wm/sim_raw.tsv" \
        "${RESULT_DIR}/clean_wm/sim_raw.tsv"
fi
printf '%s\n' "${PAIR_HASH}" > "${PAIR_HASH_FILE}"

if [[ "${STRICT_ALL_PAIRS}" == "true" ]]; then
    RESULT_DIR="${RESULT_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

coverage = json.loads((Path(os.environ["RESULT_DIR"]) / "coverage.json").read_text())
paired = coverage["paired_wav_count"]
evaluated = coverage["evaluated_common_rows"]
if paired != evaluated:
    raise RuntimeError(
        f"Not all clean/wm pairs have Seed-TTS text+prompt metadata: "
        f"paired={paired}, WER/SIM-evaluable={evaluated}. "
        "Set STRICT_ALL_PAIRS=false only if partial WER/SIM is intentional."
    )
if coverage["clean_without_wm"] or coverage["wm_without_clean"]:
    raise RuntimeError(
        "Unpaired audio exists: "
        f"clean_without_wm={len(coverage['clean_without_wm'])}, "
        f"wm_without_clean={len(coverage['wm_without_clean'])}"
    )
PY
fi
fi

QUALITY_JSON=${RESULT_DIR}/audio_quality_clean_vs_wm.json
UTMOS_JSON=${RESULT_DIR}/utmos_clean_wm.json
if [[ "${RUN_WATERMARK_QUALITY}" == "true" ]]; then
    if [[ "${FORCE_RECOMPUTE}" != "true" && -s "${QUALITY_JSON}" ]]; then
        echo "[$(date)] Audio quality already complete; skip"
    else
        echo "[$(date)] Audio quality: PESQ-WB/STOI/SI-SNR for every clean/wm pair"
        python3 batch_watermark_quality.py \
            --dir "${AUDIO_DIR}" \
            --sr "${QUALITY_SR}" \
            --json "${QUALITY_JSON}"
    fi
fi

if [[ "${RUN_UTMOS}" == "true" ]]; then
    echo "[$(date)] UTMOS: predicted naturalness for every clean/wm pair"
    utmos_cmd=(
        python3 batch_utmos.py
        --dir "${AUDIO_DIR}"
        --json "${UTMOS_JSON}"
        --repo "${UTMOS_REPO}"
        --model "${UTMOS_MODEL}"
        --device "${UTMOS_DEVICE}"
    )
    if [[ "${FORCE_RECOMPUTE}" == "true" ]]; then
        utmos_cmd+=(--force)
    fi
    "${utmos_cmd[@]}"
fi

if [[ "${RUN_WER_SIM}" == "true" ]]; then
for variant in clean wm; do
    mkdir -p "${RESULT_DIR}/${variant}"
    if [[ "${FORCE_RECOMPUTE}" != "true" && \
          -s "${RESULT_DIR}/${variant}/wer_summary.json" && \
          -s "${RESULT_DIR}/${variant}/wer_details.jsonl" ]]; then
        echo "[$(date)] WER already complete; skip: ${variant}"
    else
        echo "[$(date)] WER: ${variant}"
        python3 seed_tts_whisper_wer.py \
            --pairs "${RESULT_DIR}/${variant}_pairs.lst" \
            --output-dir "${RESULT_DIR}/${variant}" \
            --batch-size "${WHISPER_BATCH_SIZE}"
    fi
done

SV_DIR=${EVAL_REPO}/thirdparty/UniSpeech/downstreams/speaker_verification
test -f "${SV_DIR}/verification_pair_list_v2.py" || {
    echo "Incomplete official evaluator checkout: ${SV_DIR}" >&2
    exit 1
}

for variant in clean wm clean_wm; do
    mkdir -p "${RESULT_DIR}/${variant}"
    if [[ "${FORCE_RECOMPUTE}" != "true" && \
       -s "${RESULT_DIR}/${variant}/sim_raw.tsv" ]] && \
       grep -q '^avg score:' "${RESULT_DIR}/${variant}/sim_raw.tsv"; then
        echo "[$(date)] SIM already complete; skip: ${variant}"
    else
        echo "[$(date)] SIM: ${variant}"
        python3 seed_tts_run_sim.py \
            "${SV_DIR}/verification_pair_list_v2.py" \
            "${RESULT_DIR}/${variant}_pairs.lst" \
            --model_name wavlm_large \
            --checkpoint "${WAVLM_CHECKPOINT}" \
            --scores "${RESULT_DIR}/${variant}/sim_raw.tsv" \
            --wav1_start_sr 0 \
            --wav2_start_sr 0 \
            --wav1_end_sr -1 \
            --wav2_end_sr -1 \
        --device cuda:0
    fi
done
fi

WATERMARK_RESULT_DIR=${RESULT_DIR}/watermark
WATERMARK_JSON=${WATERMARK_RESULT_DIR}/watermark_summary.json
WATERMARK_TABLE=${WATERMARK_RESULT_DIR}/watermark_validation_table.txt
mkdir -p "${WATERMARK_RESULT_DIR}"

if [[ "${RUN_WATERMARK_QUALITY}" == "true" ]]; then
case "${WATERMARK_MODE}" in
    auto|reuse|recompute) ;;
    *)
        echo "WATERMARK_MODE must be auto, reuse, or recompute; got ${WATERMARK_MODE}" >&2
        exit 1
        ;;
esac

reuse_watermark=false
if [[ "${FORCE_RECOMPUTE}" != "true" && "${WATERMARK_MODE}" != "recompute" ]]; then
for candidate_dir in "${WATERMARK_RESULT_DIR}" "${AUDIO_DIR}"; do
    if [[ ! -s "${candidate_dir}/watermark_summary.json" || \
          ! -s "${candidate_dir}/watermark_validation_table.txt" ]]; then
        continue
    fi
    if CANDIDATE_DIR="${candidate_dir}" AUDIO_DIR="${AUDIO_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

candidate = Path(os.environ["CANDIDATE_DIR"])
audio_root = Path(os.environ["AUDIO_DIR"])
summary = json.loads((candidate / "watermark_summary.json").read_text())
pair_count = len(list(audio_root.glob("*_clean.wav")))
count_ok = summary.get("count") == pair_count
backend = summary.get("watermark_backend")
# Older TraceableSpeech summaries computed exact 0..15 symbol accuracy while
# labeling it Bit Acc. They must be recomputed after the binary-bit refactor.
binary_ok = backend != "traceablespeech" or (
    summary.get("bit_accuracy_unit") == "binary_bit"
    and summary.get("bit_accuracy_version", 0) >= 2
    and summary.get("bits_per_message") == 16
)
raise SystemExit(0 if count_ok and binary_ok else 1)
PY
    then
        if [[ "${candidate_dir}" != "${WATERMARK_RESULT_DIR}" ]]; then
            cp "${candidate_dir}/watermark_summary.json" "${WATERMARK_JSON}"
            cp "${candidate_dir}/watermark_validation_table.txt" "${WATERMARK_TABLE}"
        fi
        reuse_watermark=true
        echo "[$(date)] Reused complete binary-bit watermark evaluation: ${candidate_dir}"
        break
    fi
done
fi

if [[ "${WATERMARK_MODE}" == "reuse" && "${reuse_watermark}" != "true" ]]; then
    echo "WATERMARK_MODE=reuse, but no complete watermark summary/table covers all pairs." >&2
    exit 1
fi

if [[ "${reuse_watermark}" != "true" ]]; then
    echo "[$(date)] Watermark detection and attack robustness: all pairs"
    watermark_cmd=(
        python3 seed_tts_watermark_eval.py
        --audio-dir "${AUDIO_DIR}"
        --output-dir "${WATERMARK_RESULT_DIR}"
        --watermark-backend "${WATERMARK_BACKEND}"
        --audio-quality-json "${QUALITY_JSON}"
        --ts-checkpoint "${TS_CHECKPOINT}"
        --ts-config "${TS_CONFIG}"
        --voicemark-root "${VOICEMARK_ROOT}"
        --voicemark-config "${VOICEMARK_CONFIG}"
        --voicemark-st-checkpoint "${VOICEMARK_ST_CHECKPOINT}"
        --voicemark-checkpoint "${VOICEMARK_CHECKPOINT}"
    )
    if [[ "${SKIP_CODEC_ATTACKS}" == "true" ]]; then
        watermark_cmd+=(--skip-codecs)
    fi
    "${watermark_cmd[@]}"
fi
fi

summary_cmd=(
    python3 seed_tts_metric_utils.py summarize
    --output-dir "${RESULT_DIR}"
    --audio-dir "${AUDIO_DIR}"
)
if [[ "${RUN_WER_SIM}" != "true" ]]; then
    summary_cmd+=(--skip-wer-sim)
fi
if [[ "${RUN_WATERMARK_QUALITY}" == "true" ]]; then
    summary_cmd+=(
        --audio-quality-json "${QUALITY_JSON}"
        --watermark-summary-json "${WATERMARK_JSON}"
        --watermark-table "${WATERMARK_TABLE}"
    )
fi
if [[ "${RUN_UTMOS}" == "true" ]]; then
    summary_cmd+=(--utmos-json "${UTMOS_JSON}")
fi
"${summary_cmd[@]}"

cp "${RESULT_DIR}/seed_tts_clean_wm_metrics.txt" "${RESULT_DIR}/all_metrics_summary.txt"
cp "${RESULT_DIR}/seed_tts_clean_wm_metrics.json" "${RESULT_DIR}/all_metrics_summary.json"

echo "Combined table : ${RESULT_DIR}/all_metrics_summary.txt"
echo "Combined JSON  : ${RESULT_DIR}/all_metrics_summary.json"
if [[ "${RUN_WATERMARK_QUALITY}" == "true" ]]; then
    echo "Watermark table: ${WATERMARK_TABLE}"
    echo "Audio details  : ${QUALITY_JSON}"
fi
if [[ "${RUN_UTMOS}" == "true" ]]; then
    echo "UTMOS details  : ${UTMOS_JSON}"
fi
