#!/bin/bash
# High-fidelity OpenEMS validation for GA outputs.
#
# Required:
#   OPT_BASE   Root directory containing per-objective GA outputs
#   OUT_BASE   Root directory for high-fidelity validation outputs
#
# Optional:
#   PYTHON     Python executable (default: python3)
#   TOP_K      Number of candidates when evaluating --results (default: 3)
#   TASK_ID    Objective index for local runs (default: 0)
#
# Example (local):
#   OPT_BASE=./optimizer_runs OUT_BASE=./hifi_eval TASK_ID=0 ./scripts/run_hifi_eval.sh
#
# Example (SLURM):
#   sbatch --export=OPT_BASE=/scratch/$USER/optimizer_runs,OUT_BASE=/scratch/$USER/hifi_eval scripts/run_hifi_eval.sh
#
# Adapt the #SBATCH header below for your cluster policy.
#
#SBATCH --job-name=hifi_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=eval_%A_%a.out
#SBATCH --error=eval_%A_%a.err
#
# High-fidelity OpenEMS validation of GA-optimized designs.
# Each array task validates top-3 designs for one objective.
#
# Expected runtime: ~30-60 min per candidate (PML, fine mesh, 5M timesteps)
# Total: ~2-3 hours per objective
#
# Usage:
#   OPT_BASE=./optimizer_runs OUT_BASE=./hifi_eval sbatch scripts/run_hifi_eval.sh

set -euo pipefail

PYTHON="${PYTHON:-python3}"
TOP_K="${TOP_K:-3}"
OPT_BASE="${OPT_BASE:?set OPT_BASE to the optimizer results root directory}"
OUT_BASE="${OUT_BASE:?set OUT_BASE to the high-fidelity output directory}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

OBJECTIVES=(broadband_match lowpass_10ghz bandpass_5ghz notch_10ghz)
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#OBJECTIVES[@]}" ]]; then
    echo "ERROR: TASK_ID ${TASK_ID} is out of range (0..$(( ${#OBJECTIVES[@]} - 1 )))."
    exit 1
fi
OBJ="${OBJECTIVES[$TASK_ID]}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUT_BASE}/${OBJ}"

echo "=== High-Fidelity Evaluation: ${OBJ} ==="
echo "Input: ${OPT_BASE}/${OBJ}"
echo "Output: ${OUT_BASE}/${OBJ}"
echo "Node: $(hostname)"
echo "Start: $(date)"

# Find the best grid from the optimizer run
GRID="${OPT_BASE}/${OBJ}/candidate_00/pixel_grid.npy"
if [ ! -f "${GRID}" ]; then
    echo "No grid found at ${GRID}, looking for ga_results..."
    RESULTS="${OPT_BASE}/${OBJ}/ga_results.npz"
    if [ -f "${RESULTS}" ]; then
        "${PYTHON}" -m optimizer.hifi_openems \
            --results "${RESULTS}" \
            --objective "${OBJ}" \
            --output-dir "${OUT_BASE}/${OBJ}" \
            --top-k "${TOP_K}"
    else
        echo "ERROR: No optimizer results found for ${OBJ}"
        exit 1
    fi
else
    "${PYTHON}" -m optimizer.hifi_openems \
        --grid "${GRID}" \
        --objective "${OBJ}" \
        --output-dir "${OUT_BASE}/${OBJ}/candidate_00"
fi

echo "Done: $(date)"
