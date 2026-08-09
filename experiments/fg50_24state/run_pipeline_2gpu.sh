#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/fg50_24state/run_pipeline_2gpu.sh \
    --runtime /path/to/fg50_24state_runtime.npz \
    --checkpoint /path/to/physical_vit_s/best_checkpoint.pth \
    --data /path/to/cifar100 \
    [--output outputs/fg50_24state] [--gpus 0,1] \
    [--stages assignment,compensation,mc] [--mc-samples 200]

The script is resumable. Completed assignments, compensation runs, center
evaluation, and Monte Carlo seeds are reused after their integrity gates pass.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME=""
SOURCE_CHECKPOINT=""
DATA_DIR=""
OUTPUT_ROOT="${REPO_ROOT}/outputs/fg50_24state"
GPU_LIST="0,1"
STAGES="assignment,compensation,mc"
MC_SAMPLES=200
BASE_SEED=10000

while (( $# > 0 )); do
  case "$1" in
    --runtime) RUNTIME="$2"; shift 2 ;;
    --checkpoint) SOURCE_CHECKPOINT="$2"; shift 2 ;;
    --data) DATA_DIR="$2"; shift 2 ;;
    --output) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --mc-samples) MC_SAMPLES="$2"; shift 2 ;;
    --base-seed) BASE_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in RUNTIME SOURCE_CHECKPOINT DATA_DIR; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing required argument for ${value_name}" >&2
    usage >&2
    exit 2
  fi
done
for required in "${RUNTIME}" "${SOURCE_CHECKPOINT}"; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 2; }
done
[[ -d "${DATA_DIR}" ]] || { echo "Missing data directory: ${DATA_DIR}" >&2; exit 2; }

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} != 2 )); then
  echo "--gpus must contain exactly two ids, for example 0,1" >&2
  exit 2
fi
if (( MC_SAMPLES < 1 )); then
  echo "--mc-samples must be positive" >&2
  exit 2
fi

export FG50_24STATE_RUNTIME="${RUNTIME}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

has_stage() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

is_json_status() {
  local path="$1"
  local expected="$2"
  python - "${path}" "${expected}" <<'PY'
import json
import sys

try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == sys.argv[2] else 1)
PY
}

ASSIGNMENT_ROOT="${OUTPUT_ROOT}/assignment"
run_assignment() {
  local gpu="$1"
  local name="$2"
  local topk="$3"
  local tokens="$4"
  local coord_block="$5"
  local calib_batches="$6"
  local extra_flag="${7:-}"
  local out="${ASSIGNMENT_ROOT}/${name}"
  if [[ -f "${out}/assignment_complete.json" ]] && \
     is_json_status "${out}/assignment_complete.json" completed; then
    echo "[SKIP] assignment ${name}"
    return
  fi
  mkdir -p "${out}"
  local args=(
    --data "${DATA_DIR}"
    --checkpoint "${SOURCE_CHECKPOINT}"
    --output "${out}"
    --model-scale small
    --codebook-excel "${RUNTIME}"
    --codebook-sheet fg50_24state_center
    --measured-current-scale 1e-9
    --min-valid-current-na 0.39
    --assignment-q 9
    --topk "${topk}"
    --lambda-cm 0
    --assignment-chunk-devices 8192
    --calib-batches "${calib_batches}"
    --calib-batch-size 128
    --max-tokens-per-layer "${tokens}"
    --coord-block "${coord_block}"
    --sweeps 1
    --eval-batch-size 128
    --max-val-batches 0
    --num-workers 0
    --vin-lut-bins 0
    --seed 42
  )
  [[ -z "${extra_flag}" ]] || args+=("${extra_flag}")
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python "${SCRIPT_DIR}/scripts/run_assignment.py" "${args[@]}" \
    2>&1 | tee -a "${out}/console.log"
}

run_assignment_wave() {
  local pids=()
  run_assignment "${GPUS[0]}" "$1" "$2" "$3" "$4" "$5" & pids+=("$!")
  run_assignment "${GPUS[1]}" "$6" "$7" "$8" "$9" "${10}" & pids+=("$!")
  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || { echo "Assignment wave failed" >&2; exit 1; }
}

if has_stage assignment; then
  mkdir -p "${ASSIGNMENT_ROOT}"
  run_assignment "${GPUS[0]}" qpoint_source72 4 8192 64 1 --qpoint-only
  run_assignment_wave \
    a1_k64_m8192_b4 64 8192 4 1 \
    a2_k128_m8192_b4 128 8192 4 1
  run_assignment_wave \
    a3_k64_m16384_b4 64 16384 4 2 \
    a4_k128_m8192_b2 128 8192 2 1
  python "${SCRIPT_DIR}/scripts/select_best_assignment.py" \
    --assignment-root "${ASSIGNMENT_ROOT}" \
    --baseline-metrics "${ASSIGNMENT_ROOT}/qpoint_source72/metrics.json" \
    --min-accuracy 60.0 \
    --min-improvement-pp 3.0
fi

BEST_ASSIGNMENT="${ASSIGNMENT_ROOT}/a4_k128_m8192_b2/activation_aware_assignment.pt"
if [[ -f "${ASSIGNMENT_ROOT}/best_assignment.txt" ]]; then
  BEST_ASSIGNMENT="$(sed -n '1p' "${ASSIGNMENT_ROOT}/best_assignment.txt")"
fi

COMPENSATION_ROOT="${OUTPUT_ROOT}/compensation"
run_compensation() {
  local name="$1"
  local trainable="$2"
  local digital_lr="$3"
  local out="${COMPENSATION_ROOT}/${name}"
  if [[ -f "${out}/metrics.json" ]] && is_json_status "${out}/metrics.json" completed; then
    echo "[SKIP] compensation ${name}"
    return
  fi
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    torchrun --standalone --nproc_per_node=2 \
      "${SCRIPT_DIR}/scripts/run_posttrain.py" \
      --data "${DATA_DIR}" \
      --init-checkpoint "${SOURCE_CHECKPOINT}" \
      --model-scale small \
      --codebook-excel "${RUNTIME}" \
      --codebook-sheet fg50_24state_center \
      --measured-current-scale 1e-9 \
      --min-valid-current-na 0.39 \
      --assignment-q 9 \
      --lambda-cm 0 \
      --write-verify-voltage 4.1 \
      --vin-lut-bins 0 \
      --fixed-assignment "${BEST_ASSIGNMENT}" \
      --assignment-refresh-epochs 0 \
      --train-forward measured \
      --trainable "${trainable}" \
      --epochs 10 \
      --batch-size 8 \
      --num-workers 0 \
      --no-amp \
      --lr 3e-6 \
      --digital-lr "${digital_lr}" \
      --weight-decay 0.01 \
      --warmup-epochs 1 \
      --label-smoothing 0.1 \
      --kd-weight 0.5 \
      --kd-temperature 2 \
      --anchor-weight 0.03 \
      --curve-weight 0 \
      --seed 42 \
      --output "${out}" \
      --resume auto \
      2>&1 | tee -a "${out}/console.log"
}

if has_stage compensation; then
  is_json_status "${ASSIGNMENT_ROOT}/assignment_gate.json" PASS || {
    echo "Assignment gate did not pass" >&2
    exit 3
  }
  [[ -f "${BEST_ASSIGNMENT}" ]] || { echo "Missing assignment: ${BEST_ASSIGNMENT}" >&2; exit 2; }
  mkdir -p "${COMPENSATION_ROOT}"
  run_compensation b1_fc2_ln_head fc2_ln_head 1e-4
  run_compensation b2_attention_fc2_ln_head attn_fc2_ln_head 3e-5
  python "${SCRIPT_DIR}/scripts/select_best_compensation.py" \
    --compensation-root "${COMPENSATION_ROOT}" \
    --assignment-selection "${ASSIGNMENT_ROOT}/best_assignment_metrics.json" \
    --min-accuracy 65.0 \
    --min-improvement-pp 1.0
fi

BEST_CHECKPOINT="${COMPENSATION_ROOT}/b2_attention_fc2_ln_head/best_checkpoint.pth"
if [[ -f "${COMPENSATION_ROOT}/best_checkpoint.txt" ]]; then
  BEST_CHECKPOINT="$(sed -n '1p' "${COMPENSATION_ROOT}/best_checkpoint.txt")"
fi

if has_stage mc; then
  is_json_status "${COMPENSATION_ROOT}/compensation_gate.json" PASS || {
    echo "Compensation gate did not pass" >&2
    exit 3
  }
  for required in "${BEST_CHECKPOINT}" "${BEST_ASSIGNMENT}"; do
    [[ -f "${required}" ]] || { echo "Missing Monte Carlo input: ${required}" >&2; exit 2; }
  done
  MC_ROOT="${OUTPUT_ROOT}/member_mc_24state"
  mkdir -p "${MC_ROOT}"
  common=(
    --data "${DATA_DIR}"
    --checkpoint "${BEST_CHECKPOINT}"
    --fixed-assignment "${BEST_ASSIGNMENT}"
    --model-scale small
    --codebook-excel "${RUNTIME}"
    --codebook-sheet fg50_24state_center
    --member-archive "${RUNTIME}"
    --state-csv "${RUNTIME}"
    --fold-id fg50_24state
    --measured-current-scale 1e-9
    --min-valid-current-na 0.39
    --vin-lut-bins 0
    --eval-batch-size 128
    --max-val-batches 0
    --num-workers 2
  )
  if [[ ! -f "${MC_ROOT}/center/center_metrics.json" ]]; then
    mkdir -p "${MC_ROOT}/center"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
      python "${SCRIPT_DIR}/scripts/run_monte_carlo.py" "${common[@]}" \
        --output "${MC_ROOT}/center" --evaluate-center --num-seeds 0 \
        2>&1 | tee -a "${MC_ROOT}/center/console.log"
  fi
  pids=()
  for rank in 0 1; do
    count=$(( (MC_SAMPLES + 1 - rank) / 2 ))
    shard="${MC_ROOT}/shard_${rank}"
    mkdir -p "${shard}"
    CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" \
      python "${SCRIPT_DIR}/scripts/run_monte_carlo.py" "${common[@]}" \
        --output "${shard}" \
        --seed-start "$((BASE_SEED + rank))" \
        --seed-stride 2 \
        --num-seeds "${count}" \
        > >(tee -a "${shard}/console.log") 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || { echo "Monte Carlo shard failed; rerun to resume" >&2; exit 1; }
  python "${SCRIPT_DIR}/scripts/merge_mc_shards.py" \
    --input-root "${MC_ROOT}" \
    --expected-seeds "${MC_SAMPLES}" \
    --base-seed "${BASE_SEED}" \
    2>&1 | tee "${MC_ROOT}/merge.log"
fi

echo "FG50 24-state pipeline complete: ${OUTPUT_ROOT}"
