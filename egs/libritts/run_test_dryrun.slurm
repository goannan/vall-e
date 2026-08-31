#!/bin/bash
#SBATCH --job-name=eval_dryrun
#SBATCH --nodes=1
#SBATCH --nodelist=mrcompute02,mrcompute03,compute07
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

export PYENV_ROOT="/home/wu25/mrnas04home/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv shell voicemark

echo "================================================================================"
echo " Starting 2-Sample Dryrun Validation for All Evaluation Pipelines "
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU')"
echo "================================================================================"

# 1. Test AudioSeal
echo ">>> [1/4] Testing AudioSeal pipeline (2 samples)..."
python3 evaluate_external_watermark_test.py \
    --backend audioseal \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/seedTTS \
    --output-dir exp/dryrun_audioseal \
    --num-samples 2 \
    --save-audio-samples 2 \
    --device cuda:0

# 2. Test WavMark (GPU Fast)
echo ">>> [2/4] Testing WavMark pipeline (2 samples)..."
python3 evaluate_external_watermark_test.py \
    --backend wavmark \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/seedTTS \
    --output-dir exp/dryrun_wavmark \
    --num-samples 2 \
    --save-audio-samples 2 \
    --device cuda:0

# 3. Test TraceableSpeech
echo ">>> [3/4] Testing TraceableSpeech pipeline (2 samples)..."
python3 evaluate_external_watermark_test.py \
    --backend traceablespeech \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/seedTTS \
    --output-dir exp/dryrun_traceablespeech \
    --ts-checkpoint /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000 \
    --ts-config /home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json \
    --num-samples 2 \
    --save-audio-samples 2 \
    --device cuda:0

# 4. Test NeuMark Genkai v2
echo ">>> [4/4] Testing NeuMark Genkai v2 pipeline (2 samples)..."
python3 test_valle_native_watermark.py \
    --manifest /home/wu25/mrnas04home/projects/vall-e/egs/libritts/synthesized_data/seedTTS \
    --watermark-model /home/wu25/mrnas04home/projects/vall-e/egs/libritts/genkai_models/NeuMark_valle_v2.pt \
    --output-dir exp/dryrun_neumark_v2 \
    --num-samples 2 \
    --save-audio-samples 2 \
    --device cuda:0

echo "ALL 4 EVALUATION PIPELINES TESTED AND PASSED SUCCESSFULLY!"
