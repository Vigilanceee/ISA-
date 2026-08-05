#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Default: serial and resumable. Two-GPU seed parallelism:
#   ./run_5seed.sh --parallel-seeds 2
exec python3 -m isa.device_sweeps.fefet_health_runner \
  --manifest "${script_dir}/manifest.yaml" "$@"
