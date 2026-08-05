# Exact FeFET operator benchmark and Nsight profile

This experiment measures the original FeFET L-K/EKV response and its
analytical gradients.  It is an operator benchmark, not a training run.

## Backend contract

`benchmark_fefet_raw.py` imports `FeFETFunction` directly from the audited
kernel source and enforces the following settings in the recorded metadata:

- `conv_backend=reference`
- `linear_backend=reference`
- `raw_kernel_backend=legacy` for the baseline trace
- `raw_forward_backend=legacy` for the baseline trace
- `lut_enabled=false`
- `planar_enabled=false`
- `direct_conv_enabled=false`
- `surrogate_backward_enabled=false`
- `weight_quant_enabled=false`

The benchmark therefore cannot silently dispatch through an alternate device
operator.  `baseline.json` records the kernel/config SHA-256 values and the
settings before and after enforcement.

The archived baseline used kernel SHA-256
`9f01869f9afde1e80fc13d162826e068cf44c41bb5e560e78355a34e45d06102`.
The current exact kernel retains the legacy implementation behind
`raw_kernel_backend=legacy`, so the benchmark remains reproducible after the
split-reduction implementation is added.

## Baseline

Measured on an NVIDIA H100 80 GB HBM3 MIG 3g.40gb with PyTorch
2.3.0a0+6ddf5cf85e.nv24.04, CUDA 12.4, two warm-up iterations and five timed
iterations.  Shapes are exact batch-1 VGG8 im2col/linear dimensions.

| Shape | M × N × K | Forward median (ms) | Backward median (ms) | Total median (ms) |
|---|---:|---:|---:|---:|
| micro | 64 × 64 × 64 | 0.830 | 8.269 | 9.100 |
| conv1_b1 | 1024 × 128 × 27 | 0.408 | 66.533 | 66.940 |
| conv3_b1 | 256 × 256 × 1152 | 36.192 | 30.713 | 66.892 |
| fc1_b1 | 1 × 1024 × 8192 | 248.385 | 134.588 | 382.957 |

All measured outputs and gradients were finite.  A separate small-shape
PyTorch transcription of the same six-step L-K solve and EKV formula agrees
with the Triton implementation to a maximum forward relative error of 0.112%
and maximum gradient relative error of 0.414%.

## Nsight result

Nsight Systems 2024.2 captured all three expected raw kernels:

- `_fefet_fwd_kernel`: 57.278% of raw FeFET kernel time
- `_fefet_grad_v_kernel`: 23.720%
- `_fefet_grad_w_kernel`: 19.002%

Together they account for 99.932% of all CUDA kernel time in the profiled
conv3 run.  This confirms that the trace contains the original formula kernels
rather than an alternate operator.

Nsight Compute 2024.1 was also attempted, but the managed MIG instance denies
hardware performance-counter access with `ERR_NVGPUCTRPERM`.  The complete
error is preserved in `results/ncu.log`; this is an infrastructure permission
limit, not a kernel failure.

## Exact split-kernel result

The optimized backend keeps the same six-step L-K solve and EKV equations.  It
parallelizes the forward K reduction and the two analytical-gradient reductions,
using bounded FP32 partial buffers before the final output cast.  On the same
H100 MIG profile, the measured medians are:

| Shape | Legacy total (ms) | Exact split total (ms) | End-to-end speedup | Exact split forward (ms) |
|---|---:|---:|---:|---:|
| micro | 9.100 | 0.895 | 10.16x | 0.812 |
| conv1_b1 | 66.940 | 0.703 | 95.26x | 0.396 |
| conv3_b1 | 66.892 | 6.599 | 10.14x | 1.942 |
| fc1_b1 | 382.957 | 20.262 | 18.90x | 3.777 |

The split and legacy implementations have identical forward outputs in the
conv1 comparison.  At the larger conv3 and fc1 shapes, mean relative forward
differences are below 4.5e-6; their maxima occur only for reference values near
zero.  Against an independent PyTorch transcription of the original formula,
the final kernel has 0.0108% maximum relative forward error and 0.729% maximum
relative voltage-gradient error.  Every benchmarked output and gradient is
finite.

The final Nsight trace attributes 29.428% of raw-kernel time to exact forward,
35.073% to voltage gradients, and 35.500% to weight gradients.  The two-stage
forward reduction itself contributes only 0.039% of raw-kernel time.  Raw FeFET
kernels account for 99.404% of all CUDA kernel time in the profiled workload.

## Optimization priorities

1. The `split` backward backend parallelizes the legacy voltage and weight
   reductions while preserving the analytical derivatives.
2. The `split_k` forward backend evaluates exact contiguous K shards and then
   performs an FP32 reduction; small-K shapes automatically use the legacy path.
3. Add shape-aware Triton autotuning
   for block sizes and reduction split counts on the target H100 MIG profile.
4. For convolution, an exact direct-convolution kernel can remove the im2col
   materialization overhead.  It must remain a separate fidelity-preserving
   optimization and be compared against this deliberately disabled baseline.

## Reproduce

From the ISA repository root:

```bash
OUTPUT_DIR="$PWD/experiments/fefet_raw/results" \
  bash experiments/fefet_raw/run_profile.sh
```

For the historical CCI checkout, pass the source/config locations explicitly:

```bash
OUTPUT_DIR=/root/dhz/yuhuihe/ISA-results/fefet_raw_benchmark_20260805/results \
FEFET_KERNEL_FILE="/root/dhz/yuhuihe/computing primitives/devices/fefet_triton.py" \
FEFET_CONFIG_FILE="/root/dhz/yuhuihe/computing primitives/configs/device_params.yaml" \
bash experiments/fefet_raw/run_profile.sh
```

Set `RUN_NCU=1` only on a host where NVIDIA performance counters are enabled.
Set `RAW_KERNEL_BACKEND=split RAW_FORWARD_BACKEND=split_k` to profile the full
optimized exact backend; both defaults remain `legacy` so the archived baseline
cannot silently change.

## Artifacts

- `results/baseline.json`: raw forward/backward timing and formula validation
- `results/fefet_raw.nsys-rep`: Nsight Systems trace
- `results/nsys_cuda_gpu_kern_sum.csv`: exported kernel table
- `results/profile_summary.json`: raw kernel presence and time shares
- `results/ncu.log`: Nsight Compute permission diagnostic
- `results_splitk/benchmark.json`: final exact-backend timing and formula validation
- `results_splitk/nsys_cuda_gpu_kern_sum.csv`: final Nsight kernel table
- `results_splitk/profile_summary.json`: final kernel presence and time shares
