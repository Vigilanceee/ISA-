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

## Multi-device physical operators

The MLP/VGG8 study includes five device equations and selects a device-specific
physical-response operator for ReRAM, PCM, STT, FeFET, and Flash transistor.
The direct FeFET operator retains the original L-K mapping and EKV channel
response, with analytical gradients through the implicit L-K solve.

ReRAM and STT use algebraically exact factorizations. PCM, FeFET, and Flash use
exact fused formula kernels. Profiling showed that a direct strided convolution
kernel was slower than `unfold` followed by a coalesced exact matrix kernel on
the target H100 MIG, so the latter remains the training default. PCM and FeFET
parallelize their long forward and backward reductions with split-K and split
reduction. Flash reuses that skeleton after applying the exact Flash limit
`A_lk=0, B_lk=1`; this preserves the Flash EKV equation. The exact Transformer
FFN tile is retained as a validated comparison, but it is not the
fastest VGG8 training route. The default device sweep evaluates the physical
formulas directly.

Published checkpoint configurations also select LUT or polynomial-response
operators. The FeFET surrogate-gradient checkpoints retain the fitted forward
table but use the configured surrogate derivative and fixed gate calibration.
The selected PCM checkpoints include their state mapping and sinh expansion;
Flash VGG8 includes its fitted polynomial operator. These settings must be
preserved for inference and retraining. See
[the device checkpoint guide](reproducing_device_studies.md) for the exact
package-to-operator mapping and verification results.

These device-study operators and the Transformer FFN LUT kernel are independent
implementation layers with separate physical parameters, configurations, and
optimization paths.
