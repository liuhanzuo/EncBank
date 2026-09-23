#!/bin/bash
#SBATCH --job-name=qencbank-tb-instance-probe
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/srv/encbank/qencbank_runtime_20260911/server_control_20260920/instance-probe-%j.log
set -euo pipefail
umask 077
ROOT=/srv/encbank/qencbank_runtime_20260911/server_control_20260920
"$ROOT/harbor_env/bin/python" "$ROOT/instance_probe.py"
