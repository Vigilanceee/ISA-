# Kernel implementation

## Transformer FFN

The Hybrid and Physical Transformer FFNs use the fitted Flash-transistor EKV
model. The current depends on the gate/threshold difference
`ΔV = VGS - Vth`, so the expensive nonlinear portion is tabulated in a
one-dimensional LUT.

The optimized path consists of:

1. voltage mapping and threshold clamping;
2. a one-dimensional EKV LUT;
3. fused Triton forward reduction;
4. a shared-LUT CUDA backward extension;
5. TIA scaling and signed output clamping.

Vision defaults to an 8192-entry LUT. Language defaults to 4096 entries. Both
use a Triton reduction block size of eight along the input dimension.

## Multi-device approximation

The MLP/VGG8 study includes five device equations. PCM, FeFET, and Flash use a
low-rank approximation of the state/voltage response surface. ReRAM and STT use
factorized kernels and do not use the low-rank backend.

The low-rank device-study backend and the Transformer FFN LUT kernel are
separate implementations with separate configuration.
