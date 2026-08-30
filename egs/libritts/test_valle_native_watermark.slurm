#!/bin/bash
#SBATCH --job-name=eval_wm_test
#SBATCH --nodes=1
#SBATCH --nodelist=mrcompute02,mrcompute03,compute07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORK_DIR}"
mkdir -p logs

if [ -d "/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin" ]; then
  export PATH="/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin:$PATH"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate voicemark || true
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4

MANIFEST_PATH="${1:-data/tokenized_voicemark/cuts_test_valle_native.jsonl.gz}"
CKPT_PATH="${2:-exp/tts_native_neumark/20260829-003157/NeuMark_step_0020000_epoch_001.pt}"
OUTPUT_DIR="${3:-exp/eval_test_step_20000_full}"

echo "=========================================================="
echo " Starting Full VALL-E Native Watermark Test Evaluation   "
echo " Time:       $(date)                                      "
echo " Node:       $(hostname)                                  "
echo " GPU:        ${CUDA_VISIBLE_DEVICES:-auto}                "
echo " Manifest:   ${MANIFEST_PATH}                             "
echo " Checkpoint: ${CKPT_PATH}                                 "
echo " Output Dir: ${OUTPUT_DIR}                                "
echo "=========================================================="

python3 test_valle_native_watermark.py \
    --manifest "${MANIFEST_PATH}" \
    --watermark-checkpoint "${CKPT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples -1 \
    --save-audio-samples 20 \
    --device cuda:0

echo "=========================================================="
echo " Test Benchmark Finished at $(date)                       "
echo "=========================================================="
