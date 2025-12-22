#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

bash prepare.sh --stage -1 --stop-stage 3