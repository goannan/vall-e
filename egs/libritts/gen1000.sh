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

# 执行推理
python3 bin/infer_valle.py --output-dir infer/gen100 \
    --checkpoint /home/wu25/mrnas04home/projects/vall-e/egs/libritts/exp/valle/epoch-40.pt \
    --audio-prompts ./prompts/8455_210777_000067_000000.wav \
    --text-file data/texts1000.txt \
    --ts-enable true \
    --ts-checkpoint-file ./traceableSpeech/g_00150000 \
