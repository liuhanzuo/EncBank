#!/bin/bash
set -euo pipefail
exec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u /srv/encbank/qencbank_align_codex_20260911/hidden_reader_four_distill_20260921/launch.py residual_lr1e5
