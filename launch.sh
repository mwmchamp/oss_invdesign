#!/bin/bash
# Launch the RF Inverse Design GUI.
#
# Environment overrides (all optional):
#   INVDESIGN_CKPT        Surrogate checkpoint (default: ./checkpoints/best_model.pt)
#   INVDESIGN_DATASET     Dataset directory for active learning
#   INVDESIGN_LLAMA_DIR   Llama 3.2 3B Instruct directory (optional)
#   OPENEMS_LIB           OpenEMS shared-library directory (only needed if
#                         libCSXCAD/libopenEMS aren't on the default loader path)
#
# Usage:
#   ./launch.sh [--port 7860] [--share]
#
# A command-line --checkpoint flag wins over INVDESIGN_CKPT (argparse takes
# the last value).
set -eu

if [[ -n "${OPENEMS_LIB:-}" ]]; then
    export LD_LIBRARY_PATH="${OPENEMS_LIB}:${LD_LIBRARY_PATH:-}"
fi

CKPT="${INVDESIGN_CKPT:-./checkpoints/best_model.pt}"

exec python3 -m frontend.app --checkpoint "$CKPT" "$@"
