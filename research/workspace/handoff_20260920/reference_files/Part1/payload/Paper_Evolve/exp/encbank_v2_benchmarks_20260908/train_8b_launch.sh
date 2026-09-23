#!/usr/bin/env bash
set -euo pipefail
# Examples and allocation rules are in train_8b_RECIPE.md. No default GPU.
ENCBANK_TRAIN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ENCBANK_PYTHON:-python}" "${ENCBANK_TRAIN_SCRIPT_DIR}/train_8b_baseline.py" "$@"
