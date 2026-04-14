#!/bin/bash
# Train the SParamEM surrogate with the Karahan recipe (MAE + log-mag + phase,
# 0.5 dropout, delayed tanh head). Runs on a single GPU.
#
# Required environment variables (or override on the CLI):
#   INVDESIGN_DATASET     Path to a pixel-grid dataset directory (use one or
#                         more space-separated paths when invoking directly).
#   RUN_DIR               Output directory for checkpoints + logs.
#
# Optional:
#   PYTHON   Python executable (default: python3)
#
# Example:
#   INVDESIGN_DATASET=/data/pixelgrid RUN_DIR=./runs/karahan ./scripts/train_surrogate.sh
#
# Adapt the #SBATCH header below for your cluster if submitting via sbatch.
#
#SBATCH --job-name=surrogate_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00

set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATASET="${INVDESIGN_DATASET:?set INVDESIGN_DATASET to the dataset directory}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the output directory}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}"
echo "Dataset: ${DATASET}"
echo "Output:  ${RUN_DIR}"
echo "Start:   $(date)"

${PYTHON} surrogate/train.py \
    --dataset-dir ${DATASET} \
    --model-version v4 \
    --loss mae \
    --dropout 0.5 \
    --output-activation none \
    --epochs 400 \
    --batch-size 256 \
    --lr 1e-3 \
    --base-channels 64 \
    --finetune-epochs 50 \
    --finetune-lr 1e-4 \
    --finetune-output-activation tanh \
    --early-stop-patience 40 \
    --seed 1 \
    --save-dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "Done: $(date)"
