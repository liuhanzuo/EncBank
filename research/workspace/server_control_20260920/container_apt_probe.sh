#!/bin/bash
set -uo pipefail
ROOT=/srv/encbank/qencbank_runtime_20260911/server_control_20260920
export APPTAINER_TMPDIR="$ROOT/tmp" APPTAINER_CACHEDIR="$ROOT/apptainer_cache" TMPDIR="$ROOT/tmp"
apptainer exec --no-mount home,tmp,bind-paths --pwd /app --writable-tmpfs --fakeroot --containall --pid "$ROOT/sif/alexgshaw_cancel-async-tasks_20251031.sif" bash -c '
set -e
mkdir -p /tmp/encbank-apt/partial
printf "APT::Sandbox::User \"root\";\nDir::Cache::archives \"/tmp/encbank-apt\";\n" > /tmp/encbank-apt.conf
export APT_CONFIG=/tmp/encbank-apt.conf
timeout 45 apt-get update -qq
timeout 120 apt-get install -y -qq python3 python3-venv tmux asciinema
echo INSTALL_COMPLETE
/usr/bin/python3 --version
tmux -V
asciinema --version
'
