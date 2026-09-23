#!/bin/bash
set -euo pipefail
exec /srv/encbank/Paper_Evolve/.venv/bin/python -B -u /srv/encbank/qencbank_align_codex_20260911/hidden_reader_localanchors_20260921/launch.py dual_local
