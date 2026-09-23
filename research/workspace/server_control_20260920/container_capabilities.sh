#!/bin/bash
set -uo pipefail
ROOT=/srv/encbank/qcomem_runtime_20260911/server_control_20260920
export APPTAINER_TMPDIR="$ROOT/tmp"
export APPTAINER_CACHEDIR="$ROOT/apptainer_cache"
export TMPDIR="$ROOT/tmp"
apptainer exec --no-mount home,tmp,bind-paths --pwd /app --writable-tmpfs --fakeroot --containall --pid "$ROOT/sif/alexgshaw_cancel-async-tasks_20251031.sif" bash -c '
echo IDENTITY
id
echo PYTHON
command -v python3
python3 --version
echo SWITCH_USER
su nobody -s /bin/sh -c "id -u"
echo NETWORK
readlink /proc/self/ns/net
echo PACKAGE_INDEX
timeout 45 apt-get update
echo INSTALL_PYTHON
timeout 45 apt-get install -y python3 python3-venv
echo TOOLS
command -v tmux
command -v asciinema
echo DONE
'
echo HOST_NETWORK
readlink /proc/self/ns/net
