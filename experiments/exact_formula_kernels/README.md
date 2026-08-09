# Exact device-kernel profiling

This experiment profiles the formula-exact MLP/VGG8 operators. ReRAM and STT
use algebraically exact factorizations; PCM, FeFET, and Flash compare the
`unfold + exact matrix formula` implementation with the direct sliding-window
Triton convolution.

Run the correctness test before profiling:

```bash
pytest -q tests/test_exact_direct_conv.py
```

Generate a compact timing report and an Nsight Systems trace:

```bash
python experiments/exact_formula_kernels/profile_exact_conv.py \
  --output artifacts/exact_formula_kernels/benchmark.json

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output artifacts/exact_formula_kernels/nsys_exact_conv \
  python experiments/exact_formula_kernels/profile_exact_conv.py \
  --profile-pass --output artifacts/exact_formula_kernels/nsys_pass.json
```

The profile pass emits NVTX ranges for both exact routes and keeps the physical
formula backend selected throughout.

Profile the final selected VGG8 route after tuning:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output artifacts/exact_formula_kernels/nsys_vgg8_flash_selected \
  python experiments/exact_formula_kernels/profile_selected_vgg8.py \
  --device flash \
  --output artifacts/exact_formula_kernels/vgg8_flash_selected.json
```

## Selected implementation

Nsight measurements on an H100 80 GB MIG 3g.40gb instance selected the
coalesced matrix route for convolution. A direct sliding-window kernel is kept
for correctness and future tuning, but is disabled by default because its
high-fan-in VGG8 layers were slower. The Transformer FFN tile showed the same
shape dependence, so it is not used as a universal convolution backend.

The final exact operators use split-K forward reduction and split gradient
reduction where the nonlinear formula prevents algebraic factorization:

| Device | Legacy VGG8 step (ms) | Selected step (ms) | Speedup |
| --- | ---: | ---: | ---: |
| PCM | 234.14 | 74.93 | 3.12x |
| Flash | 1103.48 | 209.55 | 5.27x |

FeFET retains its formula-specialized split implementation. Its selected VGG8
step is 440.50 ms at batch size 8. ReRAM and STT remain exact factorizations and
do not need a nonlinear split reduction.
