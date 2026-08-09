#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
exec python3 experiments/exact_formula_sweep/run_sweep.py --parallel-gpus 2 "$@"
