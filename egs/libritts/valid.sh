#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

exp_dir=exp/valle
# step3 inference
python3 bin/infer.py --output-dir infer/demos \
    --checkpoint=${exp_dir}/best-valid-loss.pt \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \
    --ts-enable true \
    --ts-checkpoint-file /home/wu25/mrnas04home/projects/TraceableSpeech/save_model320/g_00150000 \
    --ts-sample-num 5 \
    --ts-bit-num 4 \