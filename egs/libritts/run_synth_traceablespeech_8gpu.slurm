#!/bin/bash
#SBATCH --job-name=synth_ts_valle_8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --output=logs/synth_ts_valle_8gpu_%j.out
#SBATCH --error=logs/synth_ts_valle_8gpu_%j.err

echo "================================================================================"
echo " Starting 8-GPU TraceableSpeech + VALL-E Parallel Synthesis & Evaluation"
echo " Date: $(date)"
echo " Host: $(hostname)"
echo " Job ID: $SLURM_JOB_ID"
echo " CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

mkdir -p logs
mkdir -p exp/valle_traceablespeech_synthesis

# Detect available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs on node $(hostname):"
nvidia-smi -L

# Setup GPU IDs string (0,1,2,3,4,5,6,7)
if [ "$NUM_GPUS" -ge 8 ]; then
    GPU_IDS="0,1,2,3,4,5,6,7"
elif [ "$NUM_GPUS" -ge 4 ]; then
    GPU_IDS="0,1,2,3"
else
    GPU_IDS="0"
fi

export PYTHONUNBUFFERED=1
PYTHON_EXEC="/home/wu25/mrnas04home/.pyenv/versions/voicemark/bin/python3"

echo ""
echo ">>> STEP 1: Synthesizing Tokens and Audio on $NUM_GPUS GPUs (LibriTTS & SeedTTS) <<<"
$PYTHON_EXEC synthesize_valle_traceablespeech_8gpu.py \
    --dataset both \
    --output-base-dir exp/valle_traceablespeech_synthesis \
    --valle-checkpoint /home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt \
    --ts-checkpoint /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000 \
    --ts-config /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json \
    --gpus "$GPU_IDS" \
    --top-k -100 \
    --temperature 1.0

SYNTH_EXIT_CODE=$?
if [ $SYNTH_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Synthesis failed with exit code $SYNTH_EXIT_CODE!"
    exit $SYNTH_EXIT_CODE
fi

echo ""
echo ">>> STEP 2: Running Full Benchmark Evaluation on Newly Synthesized TraceableSpeech Pairs <<<"

# 2.1 Evaluate LibriTTS (2,078 cuts)
echo "--- Evaluating LibriTTS (2,078 cuts) ---"
$PYTHON_EXEC evaluate_traceablespeech_valle_native.py \
    --audio-dir exp/valle_traceablespeech_synthesis/libriTTS \
    --output-dir exp/eval_libri_traceablespeech_valle_native_8gpu \
    --ts-checkpoint /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000 \
    --ts-config /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json \
    --wavlm-checkpoint models/wavlm_large_finetune.pth \
    --device cuda:0

# 2.2 Evaluate SeedTTS (1,088 cuts)
echo "--- Evaluating SeedTTS (1,088 cuts) ---"
$PYTHON_EXEC evaluate_traceablespeech_valle_native.py \
    --audio-dir exp/valle_traceablespeech_synthesis/seedTTS \
    --output-dir exp/eval_seed_traceablespeech_valle_native_8gpu \
    --ts-checkpoint /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000 \
    --ts-config /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json \
    --wavlm-checkpoint models/wavlm_large_finetune.pth \
    --device cuda:0

echo ""
echo "================================================================================"
echo " 8-GPU Synthesis & Evaluation Pipeline Finished Successfully at $(date)"
echo "================================================================================"
