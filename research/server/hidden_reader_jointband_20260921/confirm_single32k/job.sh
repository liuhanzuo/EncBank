#!/bin/bash
set -euo pipefail
exec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u /srv/encbank/qcomem_align_codex_20260911/hidden_reader_jointband_20260921/launch.py confirm_single32k
