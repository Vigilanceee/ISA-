#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${OUTPUT_DIR:-${script_dir}/results}"
python_bin="${PYTHON_BIN:-python3}"
kernel_file="${FEFET_KERNEL_FILE:-${script_dir}/../../src/isa/kernels/device_sweep/fefet_triton.py}"
config_file="${FEFET_CONFIG_FILE:-${script_dir}/../../configs/device_sweeps/device_params.yaml}"
raw_kernel_backend="${RAW_KERNEL_BACKEND:-legacy}"
raw_forward_backend="${RAW_FORWARD_BACKEND:-legacy}"

mkdir -p "${output_dir}"

"${python_bin}" "${script_dir}/benchmark_fefet_raw.py" \
  --kernel-file "${kernel_file}" \
  --config "${config_file}" \
  --raw-kernel-backend "${raw_kernel_backend}" \
  --raw-forward-backend "${raw_forward_backend}" \
  --output "${output_dir}/baseline.json" \
  --shapes "${BENCHMARK_SHAPES:-micro,conv1_b1,conv3_b1,fc1_b1}" \
  --warmup "${BENCHMARK_WARMUP:-2}" \
  --iters "${BENCHMARK_ITERS:-5}"

if ! command -v nsys >/dev/null 2>&1; then
  printf 'Nsight Systems is not installed; baseline is available at %s\n' "${output_dir}/baseline.json" >&2
  exit 3
fi

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output "${output_dir}/fefet_raw" \
  "${python_bin}" "${script_dir}/benchmark_fefet_raw.py" \
    --kernel-file "${kernel_file}" \
    --config "${config_file}" \
    --raw-kernel-backend "${raw_kernel_backend}" \
    --raw-forward-backend "${raw_forward_backend}" \
    --output "${output_dir}/nsys_profile_run.json" \
    --shapes "${PROFILE_SHAPE:-conv3_b1}" \
    --warmup 1 \
    --iters 1 \
    --skip-validation

nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  "${output_dir}/fefet_raw.nsys-rep" \
  > "${output_dir}/nsys_cuda_gpu_kern_sum.csv"

"${python_bin}" "${script_dir}/summarize_nsys.py" \
  "${output_dir}/nsys_cuda_gpu_kern_sum.csv" \
  --output "${output_dir}/profile_summary.json"

if [[ "${RUN_NCU:-0}" == "1" ]]; then
  if ! command -v ncu >/dev/null 2>&1; then
    printf 'Nsight Compute requested but ncu is not installed\n' >&2
    exit 4
  fi
  ncu \
    --force-overwrite \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name 'regex:fefet_(fwd|grad_v|grad_w)' \
    --section LaunchStats \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --export "${output_dir}/fefet_raw_ncu" \
    "${python_bin}" "${script_dir}/benchmark_fefet_raw.py" \
      --kernel-file "${kernel_file}" \
      --config "${config_file}" \
      --raw-kernel-backend "${raw_kernel_backend}" \
      --raw-forward-backend "${raw_forward_backend}" \
      --output "${output_dir}/ncu_profile_run.json" \
      --shapes "${NCU_SHAPE:-micro}" \
      --warmup 1 \
      --iters 1 \
      --skip-validation
fi

printf 'FeFET raw baseline: %s\n' "${output_dir}/baseline.json"
printf 'Nsight kernel summary: %s\n' "${output_dir}/profile_summary.json"
