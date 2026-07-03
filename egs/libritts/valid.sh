#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

exp_dir=exp/valle
# step3 inference
python3 bin/infer.py --output-dir infer/demos2 \
    --checkpoint=${exp_dir}/epoch-40.pt \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8455_210777_000067_000000.wav \
    --text-file data/texts100.txt \
    --ts-enable true \
    --ts-checkpoint-file ./traceableSpeech/g_00150000 \