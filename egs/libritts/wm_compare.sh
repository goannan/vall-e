#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

N_SAMPLES=10 OUT_DIR=infer/wm_eval_small ./run_watermark_batch.sh