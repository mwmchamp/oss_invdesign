#!/bin/bash
# Generate a pixel-grid dataset with OpenEMS.
#
# Required:
#   OUTPUT_DIR   Root directory for generated data + logs
#
# Optional:
#   PYTHON          Python executable (default: python3)
#   TOTAL_DESIGNS   Total dataset size (default: 40000)
#   TASKS           Number of array tasks (default: 2000)
#   TASK_ID         Task index for local runs (default: 0)
#
# Example (local):
#   OUTPUT_DIR=./datasets/pixelgrid TASKS=1 TOTAL_DESIGNS=20 ./scripts/generate_dataset.sh
#
# Example (SLURM):
#   sbatch --export=OUTPUT_DIR=/scratch/$USER/pixelgrid_dataset scripts/generate_dataset.sh
#
# Adapt the #SBATCH header below for your cluster policy.
#
#SBATCH --job-name=pixgv4
#SBATCH --partition=cpu
#SBATCH --array=0-1999
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#
# Dataset v4: PML boundaries, port connectivity fix.
#
# Key changes from v3:
#   - PML absorbing boundaries (was PEC) — eliminates cavity resonance artifacts
#   - Air thickness 300 µm (was 100 µm) — buffer between material and PML
#   - -35 dB end criteria — stops before PML late-time instability onset (~120K ts)
#   - 200 µm XY padding — more space for PML absorbers
#   - mesh_div 20 — IHP standard (lambda_eff / 20)
#
# Same as v3: variable fill 10-90%, randomized port placement, connectivity fix.
# Seeds 60000-99999 (non-overlapping with v1-v3).
#
# 40,000 designs across 2000 SLURM array tasks (20 designs each).
# Each design: ~5-8min with PML -> each task: ~100-160min.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the dataset output directory}"
TOTAL_DESIGNS="${TOTAL_DESIGNS:-40000}"
TASKS="${TASKS:-2000}"
DESIGNS_PER_TASK=$((TOTAL_DESIGNS / TASKS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
SEED_START=$((60000 + TASK_ID * DESIGNS_PER_TASK))
DESIGN_ID_OFFSET=$((TASK_ID * DESIGNS_PER_TASK))

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "${OUTPUT_DIR}/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "=== Task ${TASK_ID} (v4: PML + -35dB + air=300um) ==="
echo "Designs: ${DESIGN_ID_OFFSET} to $((DESIGN_ID_OFFSET + DESIGNS_PER_TASK - 1))"
echo "Seeds:   ${SEED_START} to $((SEED_START + DESIGNS_PER_TASK - 1))"
echo "Output:  ${OUTPUT_DIR}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"

"${PYTHON}" -m invdesign dataset \
    --output "${OUTPUT_DIR}" \
    --num-designs "${DESIGNS_PER_TASK}" \
    --seed "${SEED_START}" \
    --design-id-offset "${DESIGN_ID_OFFSET}" \
    --inner-size 25 \
    --ports-per-edge 1 \
    --em-backend pixel \
    --nr-ts 500000 \
    --end-criteria-db -35.0 \
    --mesh-div 20.0 \
    --f-start 1e9 \
    --f-stop 30e9 \
    --n-freq 30 \
    --xy-pad-um 200.0 \
    --use-pml \
    --skip-png

echo "Done:    $(date)"
