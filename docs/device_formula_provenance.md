# Device-formula provenance

The default MLP/VGG8 sweep evaluates the five physical response equations
below. Currents are represented internally in microamperes.

## ReRAM

`I(V,g) = I0 exp(-g/g0) sinh(V/V0)`, with `I0=200 μA`, `g0=0.15 nm`, and
`V0=0.35 V`. The implementation factorizes voltage and state terms without
approximating the equation.

## PCM

`I(V,χ) = I0(χ) χ sinh(χV/VT)`, where
`I0(χ)=3976.6 exp(-53.6321χ) μA` and `VT=0.02585 V`. The configured state
interval is `χ∈[0.15,0.31]`.

## STT-MRAM

`I(V,φ) = G0 V(1+αV²)(1+TMR cos φ)`, with `G0=331 μS`, `α=0.79 V⁻²`, and
`TMR=1.2`. The implementation uses an algebraically exact factorization.

## FeFET

The internal voltage is obtained from
`Vext−Vth = B_lk(Vint−Vth) + A_lk(Vint−Vth)³`. The resulting internal voltage
is passed through the EKV current and velocity-saturation denominator. The
mean clean-fit coefficients are `A_lk=0.8214`, `B_lk=1.3171`, `n=1.0280`,
`I_S=6.3026 μA`, and `V_sat=5.0 V`. The eight fitted threshold states are
`[1.8711, 2.1054, 2.3445, 2.5970, 2.8471, 3.1103, 3.3727, 3.7399] V`.

## Flash transistor

The EKV current is
`I_basic = I_S[softplus((V−Vth)/(2nUT))² − softplus((V−Vth−VD)/(2nUT))²]`,
followed by `I=I_basic/(1+V/V_sat)`. Parameters are `I_S=0.75875 μA`,
`n=4.1360`, `UT=0.026 V`, `VD=0.1 V`, and `V_sat=8.2624 V`.

## Kernel boundary

The Transformer FFN implementation also uses a Flash-transistor response. Its
exact tile is numerically reusable, but profiling found mixed speed
for VGG8 shapes. The selected path instead uses the exact `A_lk=0, B_lk=1`
limit of the FeFET split-reduction skeleton, which reduces identically to the
Flash EKV equation and is faster across all profiled VGG8 layers. A dedicated
direct sliding-window candidate remains available for validation, but
profiling did not justify enabling it for training.
