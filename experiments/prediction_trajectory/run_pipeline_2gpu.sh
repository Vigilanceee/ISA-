#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --data PATH --output-root PATH [--gpus 0,1] [--smoke] [--max-train-steps N] [--max-val-steps N]"
}

data_dir=""
output_root=""
gpu_ids="0,1"
smoke=0
max_train_steps=0
max_val_steps=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      data_dir="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --gpus)
      gpu_ids="$2"
      shift 2
      ;;
    --smoke)
      smoke=1
      shift
      ;;
    --max-train-steps)
      max_train_steps="$2"
      shift 2
      ;;
    --max-val-steps)
      max_val_steps="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${data_dir}" || -z "${output_root}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -d "${data_dir}/cifar-10-batches-py" ]]; then
  echo "Existing CIFAR-10 dataset not found under ${data_dir}; refusing to download inside the job." >&2
  exit 3
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config="${repo_root}/experiments/prediction_trajectory/config.yaml"
device_config="${repo_root}/configs/device_sweeps/device_params.yaml"
mkdir -p "${output_root}/logs"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
# The production CUDA image includes an older ONNX protobuf binding that is
# imported transitively by torchvision; force its compatible pure-Python parser.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

runner=(
  python3 -u -m isa.prediction_trajectory.runner
  --config "${config}"
  --device-config "${device_config}"
  --data-dir "${data_dir}"
  --output-root "${output_root}"
  --gpus "${gpu_ids}"
)
if [[ "${smoke}" -eq 1 ]]; then
  runner+=(--smoke)
fi
if [[ "${max_train_steps}" -gt 0 ]]; then
  runner+=(--max-train-steps "${max_train_steps}")
fi
if [[ "${max_val_steps}" -gt 0 ]]; then
  runner+=(--max-val-steps "${max_val_steps}")
fi

"${runner[@]}" 2>&1 | tee -a "${output_root}/logs/matrix.log"

if [[ "${smoke}" -eq 0 ]]; then
  python3 -u -m isa.prediction_trajectory.analysis \
    --config "${config}" \
    --output-root "${output_root}" \
    2>&1 | tee -a "${output_root}/logs/analysis.log"

  python3 -u -m isa.prediction_trajectory.plot \
    --output-root "${output_root}" \
    2>&1 | tee -a "${output_root}/logs/plot.log"

  python3 - "${output_root}" <<'PY'
from pathlib import Path
import json
import sys
import time

root = Path(sys.argv[1])
required = [
    root / "analysis" / "summary.json",
    root / "figures" / "figure3b_joint_pca.pdf",
    root / "figures" / "figure3c_prediction_geometry.pdf",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("missing final artifacts: " + ", ".join(missing))
(root / "completed.json").write_text(
    json.dumps(
        {
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "required_artifacts": [str(path.relative_to(root)) for path in required],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
fi

echo "PREDICTION_TRAJECTORY_PIPELINE_DONE output=${output_root} smoke=${smoke}"
