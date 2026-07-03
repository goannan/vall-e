#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

exp_dir=${EXP_DIR:-exp/valle}
watermark_backend=${WATERMARK_BACKEND:-voicemark}
ts_enable=${TS_ENABLE:-false}
ts_ckpt=${TS_CKPT:-./traceableSpeech/g_00150000}
voicemark_root=${VOICEMARK_ROOT:-/home/wu25/mrnas04home/projects/VoiceMark}
voicemark_checkpoint=${VOICEMARK_CHECKPOINT:-train/Log/spt_base/20260601-123358/WatermarkTrainer_final_00150000.pt}
# step3 inference
python3 bin/infer.py --output-dir infer/demos2 \
    --checkpoint=${exp_dir}/epoch-40.pt \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8455_210777_000067_000000.wav \
    --text-file data/texts100.txt \
    --watermark-backend "${watermark_backend}" \
    --ts-enable "${ts_enable}" \
    --ts-checkpoint-file "${ts_ckpt}" \
    --voicemark-root "${voicemark_root}" \
    --voicemark-checkpoint "${voicemark_checkpoint}"
