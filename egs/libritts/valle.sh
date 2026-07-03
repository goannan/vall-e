#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e  # 添加这行：一旦任一命令失败，立即停止脚本运行

# 准备工作
exp_dir=exp/joint
mkdir -p ${exp_dir}/log  # 确保 joint 文件夹和日志目录存在
mkdir -p logs            # 确保 SLURM 日志目录存在

## Train AR model (Stage 1)
# 如果您是第一次运行且 joint 里还没有模型，请保持如下设置
python3 bin/joint_trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --world-size 2 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir}

## Train NAR model (Stage 2)
# 这一步需要 Stage 1 产出的 best-valid-loss.pt，如果成功了，这里就不会再报错
if [ -f "${exp_dir}/best-valid-loss.pt" ]; then
    cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt
    python3 bin/joint_trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
          --world-size 2 \
          --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
          --model-name valle --share-embedding true --norm-first true --add-prenet false \
          --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
          --base-lr 0.05 --warmup-steps 200 --average-period 0 \
          --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
          --exp-dir ${exp_dir}
else
    echo "Error: Stage 1 training failed to produce best-valid-loss.pt in ${exp_dir}"
    exit 1
fi
