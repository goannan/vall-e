#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 创建输出目录
mkdir -p infer/gen100

# 初始化 pyenv 环境
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 激活恢复正常的 valle 环境
pyenv activate valle

WATERMARK_BACKEND=${WATERMARK_BACKEND:-voicemark}
VOICEMARK_ROOT=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}
VOICEMARK_CHECKPOINT=${VOICEMARK_CHECKPOINT:-train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt}

# 执行推理
python3 bin/infer_valle.py --output-dir infer/gen100 \
    --checkpoint /home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt \
    --audio-prompts ./prompts/8455_210777_000067_000000.wav \
    --text-file data/texts1000.txt \
    --watermark-backend "${WATERMARK_BACKEND}" \
    --ts-enable false \
    --voicemark-root "${VOICEMARK_ROOT}" \
    --voicemark-checkpoint "${VOICEMARK_CHECKPOINT}"
