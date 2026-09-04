# Arm SVE2/Neon I8MM codegen results

This branch is based on the official Triton-CPU 3.7.2 commit
`3de2b2c424fd03550c1dc1efa1461ae301138e75`.  The measurements below were
collected on the CIX AArch64 machine with one thread pinned to a Cortex-A720
core.  The machine reports 128-bit SVE, SVE2, dot-product, and I8MM support.

The optimized kernels are ordinary Triton programs built from `tl.load`,
integer expressions, `tl.dot`, and `tl.store`.  They do not call TLE_raw or a
hand-written runtime compute function.  The C++ measurements call the exported
kernel symbol from the generated shared object directly, removing the Python
launcher from the comparison.

## Arm target selection

The CPU backend selects the lowering from detected host capabilities rather
than from a model or vendor name. Linux supplements LLVM host detection with
the features common to every CPU in `/proc/cpuinfo` and queries the active SVE
vector length with `PR_SVE_GET_VL`. Darwin supplements LLVM detection with
`hw.optional.arm.FEAT_DotProd` and `hw.optional.arm.FEAT_I8MM` sysctls.

An AArch64 host with SVE2, I8MM, and a 128-bit active SVE vector length uses
the scalable lowering. A host with DotProd and I8MM but no supported SVE
configuration uses the fixed-width lowering automatically. The latter is the
intended Apple Silicon path: M1 decode lowers to Neon SDOT, while packed Q4 and
Q8 prefill lower to Neon SMMLA. Generic M>=8 row-major dot candidates remain
disabled in fixed mode because their current rolled lowering is SVE-specific.
`TRITON_CPU_FIXED_I8MM=1` forces this mode for validation on an SVE host;
`TRITON_CPU_DISABLE_SVE2_I8MM=1` disables both target paths for strict A/B.
The target mode and detected feature set are part of the backend cache key.
The supplemented feature string is passed into LLVM host assembly generation,
not only used to select the MLIR pass. Fixed-mode validation also passes
explicit negative SVE/SVE2 features, preventing later LLVM vectorization from
reintroducing scalable instructions.

Darwin and Linux host JIT assembly use explicit
`-march=armv8.6-a+sve2+i8mm` or
`-march=armv8.6-a+dotprod+i8mm`, with `+fp16` and `+bf16` appended when the
OS-reported host features support them. Linux also passes `-mcpu=native`
before the backend-specific `-march` flag. This keeps LLVM's selected `.8h`
instructions consistent with the system assembler feature set. CIX
forced-fixed tests validate fixed-width
code generation and numerics, but they are not a substitute for Apple M4 Pro
latency measurements.

## Results

Lower is better.  Times are median latency for a single pinned thread.

| Kernel and shape | Triton direct | matching ACLE C | Triton / C |
| --- | ---: | ---: | ---: |
| int8 GEMM, 32x32x128 | 1.52 us | 1.86 us | 0.82x |
| int8 GEMM, 128x128x128, BM=128 | 20.5 us | 25.3 us | 0.81x |
| int8 GEMM, 256x256x256 | 164.8-165.3 us | 188.9-190.5 us | 0.87x |
| packed W8A8 GEMV, K=256, N=256 | 2.085 us | 2.284 us | 0.913x |
| packed W8A8 GEMV, K=1024, N=3072 | 104.82 us | 114.82 us | 0.913x |
| packed W8A8 GEMV, K=9728, N=2560 | 960.25 us | 935.25 us | 1.027x |
| BF16 activation -> W8 RNE, K=1024, N=3072, BN=64 | 80.16 us | 80.61 us | 0.994x |
| exact-KAI-layout W8 decode, K=1024, N=3072 | 78.17 us | KleidiAI 79.64 us | 0.982x |
| exact-KAI-layout W8 decode, K=4096, N=2560 | 265.72 us | KleidiAI 268.22 us | 0.991x |
| exact-KAI-layout W8 decode, K=2560, N=9728 | 958.80 us | KleidiAI 1013.83 us | 0.946x |
| exact-KAI-layout W8 decode, K=9728, N=2560 | 953.92 us | KleidiAI 1006.51 us | 0.948x |
| exact-KAI W8 BF16 pipeline, K=1024, N=3072 | 80.57-80.98 us | KleidiAI 81.15-81.26 us | 0.992-0.998x |
| exact-KAI W8 BF16 pipeline, K=2560, N=9728 | 983.13-990.08 us | KleidiAI 1017.98-1021.92 us | 0.962-0.973x |
| exact-KAI W8 BF16 pipeline, K=9728, N=2560 | 982.75-985.20 us | KleidiAI 1010.82-1012.32 us | 0.972-0.973x |
| W4A8 GEMV, K=256, N=256 | 3.049 us | 3.467 us | 0.879x |
| W4A8 GEMV, K=9728, N=2560 | 1.141 ms | 1.315 ms | 0.868x |
| Q4_1 x Q8_1 GEMV, K=256, N=256 | 3.334 us | 3.638 us | 0.917x |
| Q4_1 x Q8_1 GEMV, K=9728, N=2560 | 1.260 ms | 1.380 ms | 0.913x |
| KAI-layout Q4 prefill, M4, N=K=1024 | 73.55 us | KleidiAI 74.27 us | 0.990x |
| KAI-layout Q4 prefill, M8, N=K=1024 | 128.06 us | KleidiAI 147.40 us | 0.869x |
| KAI-layout Q4 prefill, M12, N=K=1024 | 183.21 us | KleidiAI 220.80 us | 0.830x |
| KAI-layout Q4 prefill, M16, N=K=1024 | 238.93 us | KleidiAI 239.02 us | 1.000x |
| KAI-layout Q4 prefill, M16, N=3072, K=1024 | 717.65 us | KleidiAI 718.89 us | 0.998x |
| KAI-layout Q4 prefill, M16, N=1024, K=3072 | 716.09 us | KleidiAI 713.43 us | 1.004x |
| KAI-layout Q4 prefill, M16, N=K=4096 | 3861.06 us | KleidiAI 3843.88 us | 1.004x |
| KAI-layout Q4 prefill, fixed Neon M16, N=1024, K=256 | 61.92 us | KleidiAI 61.58 us | 1.006x |
| KAI-layout Q4 prefill, fixed Neon M16, N=K=1024 | 230.99 us | KleidiAI 229.85 us | 1.005x |
| KAI-layout Q4 prefill, fixed Neon M16, N=3072, K=1024 | 721.53 us | KleidiAI 717.66 us | 1.005x |
| KAI-layout Q4 prefill, fixed Neon M16, N=1024, K=3072 | 692.80 us | KleidiAI 686.27 us | 1.010x |
| KAI-layout Q4 pack+prefill, M4, N=K=1024 | 75.27 us | KleidiAI components 75.50 us | 0.997x |
| KAI-layout Q4 pack+prefill, M8, N=K=1024 | 131.23 us | KleidiAI components 150.91 us | 0.870x |
| KAI-layout Q4 pack+prefill, M12, N=K=1024 | 188.10 us | KleidiAI components 226.51 us | 0.830x |
| KAI-layout Q4 pack+prefill, M16, N=K=1024 | 245.79 us | KleidiAI components 246.93 us | 0.995x |
| KAI-layout W8 activation pack, M16, K=1024 | 8.42 us | ACLE Neon 12.37 us | 0.681x |
| KAI-layout W8 prefill, M16, N=K=1024 | 115.45 us | KleidiAI 113.41 us | 1.018x |

The C implementations use Arm ACLE dot-product intrinsics and exactly the same
packed layouts as the Triton kernels.  They are useful codegen ceilings, not a
claim that every production library uses the same packing or blocking.

## Strict codegen A/B

`TRITON_CPU_DISABLE_SVE2_I8MM=1` disables the new lowering while keeping the
same Triton source and input data.

| Kernel | optimized | generic lowering | speedup |
| --- | ---: | ---: | ---: |
| int8 GEMM, 32x32x128 | 1.52 us | 13.00 us | 8.57x |
| packed W8A8 GEMV, K=256, N=256 | 2.085 us | 33.14 us | 15.89x |
| W4A8 GEMV, K=256, N=256 | 3.05 us | 36.76 us | 12.05x |
| Q4_1 x Q8_1, K=256, N=256 | 3.334 us | 37.620 us | 11.28x |
| KAI-layout Q4 prefill, M16, N=K=1024 | 272.8 us | 2510.7 us | 9.20x |
| KAI-layout W8 prefill, M16, N=K=1024 | 115.45 us | 1944.55 us | 16.84x |

For Q4_1, the optimized assembly has eight static `sdot` instructions and 201
lines.  The generic assembly has no `sdot` and 719 lines.  Neither variant
calls an external compute runtime.

## KAI-layout Q4 prefill from ordinary Triton

The Q4 prefill source remains a regular Triton kernel.  It loads the official
KleidiAI-compatible QSI8D32P/QSI4C32P blobs, expresses nibble unpacking with
integer operations, and uses four ordinary M4 `tl.dot` panels for an M16
block.  There is no Q4 intrinsic in the frontend and no TLE_raw or runtime C
compute call.

The Arm lowering recognizes the exact packed-layout algebra after the normal
Triton-to-CPU conversion.  It keeps the K32 loop rolled, lowers each panel to
16 fixed-width Neon `SMMLA` operations, converts the scaled integer result with
the fixed-point `SCVTF #4` form, and carries FP32 row accumulators through the
loop.  Panel order is selected per target because LLVM 20's two-address
coalescer assigns the Neon mask and FP32 accumulators differently: SVE uses
natural row order, while fixed-width Neon uses reverse row order.  Both final
M16 objects have 64 static `SMMLA` instructions, no vector-to-vector rotation
moves, and no stack traffic in the hot loop.  The only stack accesses are four
callee-saved SIMD register pairs in the function prologue and epilogue.

The strict comparison uses the official KleidiAI packed blobs and calls both
compute symbols directly.  Activation packing and Python launch time are
excluded.  Triton stores BF16 while the KleidiAI microkernel stores FP32; the
harness rounds the latter to BF16 before comparison.  Every tabled shape has
zero mismatches and zero BF16 ULP distance. On the SVE path, M16 is also 4-6%
faster than launching two generated M8 blocks across the tested shapes, so the
result is not an accidental M8 decomposition.

Activation packing is ordinary generated Triton as well.  A direct generated
symbol call takes 6.66 us for M16/K1024, versus 7.56 us for the official
KleidiAI packer from the same BF16-equivalent input.  The packed blobs are byte
identical, including a finite BF16 edge-pattern suite.  The C++ two-stage
wrapper measures 245.79 us for pack plus matrix, versus 245.65 us when the
same generated symbols are timed separately.  The corresponding KleidiAI
pack-plus-matrix sum is 246.93 us, a 0.995x ratio for the complete native
compute path.  The wrapper only dispatches generated symbols and contains no
compute implementation.

The pack improvement is expressed with ordinary Triton operations.  For a
finite BF16 block, clearing the sign bit leaves a monotonic magnitude encoding.
Three lane-wise integer maxima combine the four K8 slices before one horizontal
reduction.  Quantization then uses the scale derived from the same K32 block,
so the redundant +/-127 saturation can be removed before RNE conversion.  The
final object has 12 `UMAX` and 32 `FCVTNS` instructions, no `FMAXNM`/`FMINNM`
quantization clamps, external call, or spill.  M4/M8/M12/M16 direct comparisons
at K=256, 1024, 3072, and 4096 are 0.876-0.883x the KleidiAI pack latency, all
with byte-identical blobs.  The masked partial-panel path retains the original
floating-point reduction.

The M1-M3 compact decode pack reuses the finite-BF16 lane-max reduction while
preserving the existing 34-byte-per-K32 layout. Its direct function body falls
from 107 to 75 instructions: the previous 16 `FMAXNM` and 8 `FMINNM` operations
are absent, while three `UMAX` and eight `FCVTNS` remain. There is no external
call or stack access. Paired direct calls at M1/K1024 improve from 0.808 to
0.565 us (1.43x); M2 and M3 scale linearly at the same speedup. Blobs are
byte-identical to the independent floating-point/clamped pack for normal inputs
and a finite BF16 edge suite. When followed by the unchanged generated SDOT
matrix symbol, paired M1-M3/N=K=1024 pipelines improve by about 0.5% because the
matrix loop dominates the complete decode operator.

## KAI-layout W8 prefill from ordinary Triton

The Triton-CPU 3.7.2 port now recognizes the exact M4/K8 LHS and
N4/K8 RHS algebra built from ordinary `tl.load`, `reshape`, `permute`, `join`,
and `tl.dot`.  Four physical M4 panels become one logical M16 dot in the
frontend. The lowering carries sixteen native 2x2 accumulators through the K
loop and reconstructs the logical result only once. Those accumulators are SVE
vectors on the CIX path and fixed 128-bit Neon vectors on the no-SVE path. Both
forms emit no TLE operation, external runtime call, or residual
`triton_cpu.dot`.

For M16/N=K=1024, the disabled-lowering object has 1441 assembly lines, no
SMMLA, and 24 folded stack references.  The optimized object has 491 lines,
64 static SVE SMMLA instructions, and no folded spill/reload.  Direct calls to
the same generated symbol improve from 1944.55 us to 115.45 us (16.84x).
KleidiAI v1.24 takes 113.41 us in the same alternating-order harness, so the
generated loop is within 1.8% of the native library.  Packing and Python are
excluded; both kernels consume the same official KAI blobs.  Maximum FP32
absolute error is 2.38e-7.

The direct Triton/KleidiAI ratios remain 1.017x at N3072/K1024, 1.003x at
N1024/K3072, and 1.014x at N=K=4096, with the same 2.38e-7 maximum error.
This checks both N-heavy and K-heavy shapes.  The production BF16-output
route also has dedicated M4, M8, and fused M8+M4 M12 tails.  At
M=4/8/12/16 and N=K=1024 their matrix objects contain 16/32/48/64 SMMLA
instructions, 163/229/352/433 assembly lines, no external calls, and no
folded spill/reload.  Including activation packing, allocation, and Python
dispatch, these routes are respectively 2.29x, 1.91x, 2.89x, and 2.52x faster
than the retired row-major Triton path; every output is bit-exact.

The regression suite also compiles a one-permutation-different RHS graph.  It
checks the independent numerical result and requires that graph to stay on
the generic path.  This guards the physical-layout matcher against accepting
an ABI it cannot prove.

The same independent test runs the fixed lowering at K=256, 1024, and 3072.
Every int32 result is exact. Each M16 object contains 64 Neon SMMLA calls, no
SVE intrinsic, no folded spill/reload, no external call, and no residual dot.
A same-process C++ harness loads the SVE and fixed shared objects together,
alternates their direct symbol calls, and requires bit-exact outputs. With
packing and Python excluded, SVE/fixed latencies are 115.038/114.901 us at
N=K=1024, 329.261/329.516 us at N3072/K1024, and 328.989/334.150 us at
N1024/K3072. Fixed Neon is therefore within 0.2% for the square and N-heavy
shapes and 1.57% slower for the K-heavy shape on CIX. This validates the
rolled microkernel structure; it does not establish Apple performance.
Direct fixed-Neon/KleidiAI ratios are 1.017x at N=K=1024 and 1.013x at
N1024/K3072, with 2.38e-7 maximum FP32 absolute error.

The fixed path also passes the production FlagGems router for all
M=1/2/3/4/7/8/12/15/16/17/20/24/28/31/32/47/48 routes at K=1024, including
the M4/M8/M12 tails. It passes the vLLM loader slot, bias, direct eager
execution, dynamic shapes, and `torch.compile(..., fullgraph=True)`. All
tested Q4 outputs remain within four BF16 ULP of the independent reference and
the legacy row pack is never selected.

The original fixed-width natural-order M16 suffered a 30-instruction vector
register rotation and one hot spill/reload pair. Reversing only the four
independent M4 panel updates removes every rotation move and leaves eight
prologue/epilogue stack references, matching the SVE code-quality bound.
Same-process paired tests at M=16/32/48 take 265.24/521.60/751.20 us for M16
and 280.52/551.20/795.21 us for two M8 tiles, a consistent 5.7-5.9% M16
advantage. Direct C++ calls are within 0.5-1.0% of KleidiAI across K256,
N=K=1024, N3072/K1024, and N1024/K3072. Fixed and SVE production routers now
both use M16; `FLAGGEMS_ARM_Q4_M16_AS_M8=1` retains the old split schedule for
strict A/B.

A separate C++ wrapper calls the production BF16 activation pack and matrix
symbols directly, reusing the same buffers for strict old/new pairs. At
N=K=1024, M4/M8/M12/M16 pipelines improve from
46.623/74.665/95.769/124.984 us to 46.006/74.077/93.477/122.198 us. The wrapper
contains no matrix implementation and uses no Python dispatch. Its independent
scalar reference is BF16 bit-exact for every shape.

For the pack alone, a contract-equivalent ACLE Neon implementation receives
the same BF16 input and produces a byte-identical KAI blob. Across M4-M16, the
generated pack takes 2.120/4.246/6.318/8.421 us, versus ACLE's
3.098/6.190/9.282/12.372 us (a 0.681-0.686x ratio). Compared with the previous
generated floating-point/clamped pack, the new object is 1.34x faster at every
M. It combines four finite-BF16 K8 slices with three integer lane maxima before
one horizontal reduction and removes redundant per-K8 saturation. The object
has three UMAX, two FMAXNM, no FMINNM, external call, or spill. Blobs match the
old path for finite edge values, and a scalar check covers every finite BF16
magnitude and sign. This separates generated compute cost from allocation and
launcher overhead in the Python production router.

The pack ratio to ACLE is 0.756x at K256, 0.681x at K1024, and 0.679x at both
K3072 and K4096. LLVM fully unrolls the eight K32 reductions at K256, so that
object has 24 UMAX and 9 FMAXNM sites; larger shapes retain the rolled form
with three UMAX and two FMAXNM sites. Every form has zero FMINNM, spill,
external call, or residual Triton operation. The audit gates both the compact
rolled object and the bounded K256 unrolled object.

The same two FlagGems JITFunctions now run through the CPU backend of
`libtriton_jit`, using an import-only shim rather than copying the kernels.
After warmup, M4/M8/M12/M16 raw C++ pipelines take
42.69/70.51/92.16/122.04 us, while typed API latencies are
45.55/73.41/95.04/124.77 us. A generic CPU ABI fast path handles two pointers
followed by two `i32` arguments, removing libffi calls from the activation pack
grid. All four Q8 shapes use an independent scalar reference and are BF16
bit-exact.
An independent shared-library test covers this argument and grid layout, and
the focused `libtriton_jit` CPU tests pass 3/3.

Q4 uses the same integration route without copying its FlagGems kernels.  Its
M16/N=K=1024 generated pack-plus-matrix pipeline takes 249.20 us through the
typed API and 246.35 us through the raw API, versus 245.79 us for direct symbol
dispatch.  Generic and raw C++ outputs are BF16-bit-identical; independent Q4
numerics remain covered by the port regression and KleidiAI comparison above.
The M4/M8/M12 raw pipelines take 76.82/132.76/189.46 us, with typed API
latencies of 79.46/135.58/192.36 us; every typed/raw pair is bit-identical.
Before adding generic three-pointer/two-i32 and five-pointer CPU ABI fast paths,
the raw result was 264.78 us.  Direct calls therefore remove about 15.2 us of
libffi overhead without introducing any Q4-specific runtime compute code.

## W8A8 decode path and CPU loop unrolling

The packed W8A8 kernel uses only `tl.load`, `tl.trans`, `tl.dot`, and
`tl.store`.  Each packed 16-byte tile is `[4 output lanes, 4 K lanes]`; the
Arm M=1 lowering turns two unrolled dot updates into two static SDOT
instructions in a rolled loop.

Upstream Triton-CPU 3.7.2 did not run the standard TTIR loop-unroll pass, so
`tl.range(..., loop_unroll_factor=2)` left an annotation in TTIR but did not
change generated code.  This port adds the same pass already used by the
NVIDIA and AMD backends.  Factor two reduced K=9728,N=2560 direct latency from
about 995.1 us to 960.25 us in the final clean build. Factor one, four, and
eight, plus a manually split dual-accumulator form, were all slower and are
not used.  For K=1024,N=3072 and K=256,N=256, the factor-two Triton kernel is
about 8.7% faster than the matching ACLE C loop. The longest tested K remains
2.7% slower than the GCC-built ACLE C loop and is recorded as an open
optimization target rather than reported as parity. The generated loop is at
parity with the same C source compiled by the matching LLVM toolchain, so the
remaining gap is not evidence of a hidden Triton scalar fallback.

## Exact-KAI-layout W8 decode from ordinary Triton

A second W8 decode experiment consumes KleidiAI's exact asymmetric-LHS
`qai8dxp` and N4/K8 `qsi8cxp` packed blobs, including its sums, scales, bias,
and clamp epilogue. The matrix loop is still expressed with ordinary
`tl.load`, `reshape`, `join`, `permute`, `tl.dot`, and `tl.store`; it does not
call a KleidiAI or TLE runtime compute symbol.

Preserving each activation load as a 2x4 value through `tl.join` is important.
Flattening it before the join made LLVM select one paired 64-bit load followed
by two lane-insertion moves per K8 group. Keeping the physical shape intact
selects two broadcast loads (`LD1R`) and feeds SDOT directly. A strict
same-process old/new generated-object comparison improves the long-K loop by
about 1.2%, with bit-exact generated outputs. The final unroll-two object has
16 static SDOT instructions, one ADDP, no lane-insertion moves, no stack
access, no residual Triton dot, and no external call.

Explicit pointer induction is the stable schedule over K=256 through 9728.
It beats the split-index form by about 3.6-6.1% on the small and medium test
shapes, is neutral at K4096/N2560, and remains faster at both N-heavy and
K-heavy long shapes. `auto` therefore selects the pointer form from measured
shape coverage rather than a vendor or CPU name. Direct calls are 0.982x,
0.991x, 0.946x, and 0.948x KleidiAI at the four tabled shapes; pack time and
Python launch are excluded, and maximum FP32 absolute error is at most
2.39e-7.

For the production BF16 contract, a second ordinary Triton kernel implements
KAI's asymmetric qai8dxp activation pack. Its MR1 and MR4 blobs are byte-exact
with the official BF16 packer. Although pack alone is 0.786 versus 0.604 us at
M1/K1024, the generated matrix stores BF16 directly. Complete pack, matrix,
and BF16-output pipelines are 0.992-0.998x, 0.962-0.973x, and 0.972-0.973x
KleidiAI at the three
BF16 table shapes, with bit-exact final output. This prevents a matrix-only
microbenchmark from hiding a pack or conversion regression.

## Ordinary BF16-activation W8 split

The model-facing decode route has two ordinary Triton stages: a BF16 to
dynamic INT8 quantizer followed by the packed W8 `tl.dot` kernel with a BF16
store.  A C++ wrapper calls both generated symbols directly and owns the INT8
scratch vector.  All comparisons above are bit-exact for the quantized
activation, scale, and BF16 output.

The model-compatible quantizer uses round-to-nearest-even and the same
activation-scale then weight-scale multiplication order as the existing TLE
path. For K=1024,N=3072, a 64-output ordinary `tl.dot` kernel plus quantizer
takes 80.16 us through direct symbols, while a matching fused ACLE C
implementation takes 80.61 us. A single ctypes call through the C++ dispatcher
takes 83.21 us. This is a dispatcher around generated symbols, not a runtime C
compute implementation.

The same graph is now the active FlagGems JIT fallback in the independent
3.7.2 port, not only a benchmark/AOT source. `_w8_decode_sdot_kernel` supports
one rolled whole-projection program and an N64 program grid. W8 Linear, fused
QKV, and fused MLP use it whenever an optional AOT bundle is absent; the port
can import and execute all three with the development-only TLE frontend
unavailable. Fused QKV performs one shared quantization and one joined matrix
kernel. Fused MLP deliberately remains three register-light stages:
quantization, joined gate/up SDOT, and shape-specialized SwiGLU.

At K=N=1024, both whole-projection and grid objects have 32 static SDOT
instructions, no spill/reload, no stack reference, no residual dot, and no
external compute call. The whole/grid objects are 285/281 assembly lines.
Shape specialization first removes a duplicate dynamic tail from ordinary
SwiGLU. The active kernel then expresses the SLEEF-u10 exp polynomial as
ordinary Triton FMA/integer arithmetic. Its fixed/SVE-target objects are
540/526 lines, contain no external math call, and have four folded
spill/reload annotations apiece (eight bounded callee-save stack accesses).
The activation-only latency at N3072 improves from 21.299 to 18.329 us on
CPU11. All 65,280 finite BF16 gate encodings are bit-exact against
`F.silu(gate) * up` with unit `up`.

Standalone `arm_silu_and_mul` uses an `auto` threshold rather than replacing
ATen at every size. Including output allocation and Python dispatch, Triton is
still slower through N6144; at N8192 it takes 38.033 us versus ATen's
47.244 us, then 49.129 versus 69.367 us at N12288. The default threshold is
therefore 8192. `GEMS_ARM_SWIGLU_MODE=ordinary|aten` and
`GEMS_ARM_SWIGLU_MIN_ELEMENTS` retain explicit A/B control. Fused MLP invokes
the epilogue directly inside its measured three-stage route and does not use
this standalone threshold.

A fixed-target correctness run covers W8 decode at K/N = 256/256, 256/512,
256/768 (padded output), 1004/512 and 1008/512 (quantizer tails), 1024/1024,
3072/1024, and 1024/3072. Every matrix/dequant result is bit-exact against the
generated INT8 activation reference. Prefill remains bit-exact at thirteen
M values from 2 through 64. Fused MLP and both single-core and multicore QKV
are bit-exact. The same suite passes with the default SVE target and reports
`tle_decode_frontend_available=false`.

The quantizer is additionally exercised over all 65,280 finite BF16 bit
patterns in 255 independent vectors. Every activation scale is bit-exact and
there are no non-half-tie integer mismatches. Two values lie on an exact
mathematical half-tie where LLVM reassociation differs by one count from an
explicitly staged FP32 expression; both satisfy round-to-nearest-even and are
reported separately by the validator rather than hidden as generic error.

Pinned CPU11 microbenchmarks separate code generation from Python orchestration:

| Route | ordinary AOT | ordinary JIT wrapper | preallocated JIT stages |
| --- | ---: | ---: | ---: |
| W8 Linear, K=N=1024 | 48.982 us | 133.637 us | 51.026 us |
| fused QKV, K1024, N=2048+1024+1024 | 145.760 us | 220.994 us | not separately measured |
| gate/up SwiGLU, K1024, N=3072 | 194.299 us | 299.689 us | 225.010 us |

For the Linear row, the preallocated value is the two active production
Triton kernels with the same BN64/UNROLL2 graph. Thus the
large `Module.forward` gap is Python allocation and JIT dispatch composition,
not a hidden C implementation or degraded SDOT loop. The AOT wrapper removes
that orchestration while calling only generated symbols.

A one-kernel ordinary-Triton alternative was measured but rejected for the
model route. It computes absmax once, then performs RNE directly inside each
output-block dot to remove the INT8 scratch and one launch. At K1024/BN64 its
fused/split ratios are 0.843x for N256, 1.155x for N512, 1.372x for N1024,
and 1.635x for N3072. BN32 improves N256 to 0.749x but is 1.006x at
N512. Repeating conversion for every output block, plus one spill/reload pair,
dominates normal decoder dimensions. The candidate remains a benchmark rather
than a broad production heuristic.

The active JIT fallback now shares the AOT tile selector. Decoder, QKV and MLP
projections use BN64, while vocabulary projections with N>=32768 use BN32. In
an alternating direct-kernel comparison at N=152064/K=1024, BN32 takes
7156.51 us versus 7533.41 us for BN64 (0.950x) with bit-exact output. The
BN32 object has 16 static SDOT instructions, 200 assembly lines, no
spill/reload, external call, or residual dot. This closes a former gap where a
missing AOT bundle silently selected the slower vocabulary tile.

The quantizer deliberately uses `BLOCK_K=16`. For each full K32 group it
clears the finite BF16 sign bit, computes three lane-wise integer maxima over
four K8 slices, and performs one horizontal floating-point reduction. LLVM
then combines round-even and integer conversion into four vector `FCVTNS`
instructions. The K1024 object has 211 assembly lines, three `UMAX`, two
`FMAXNM`, no external call, and no stack traffic. A masked floating-point tail
preserves K values divisible by 16 but not 32; K1008 and K1024 both pass
finite-BF16 edge-pattern and independent scalar reference checks.

In paired direct calls at K1024, replacing the retired manual-RNE quantizer
takes latency from 1.512 to 0.349 us (4.33x). Isolating only the absmax change
against the same native-RNE implementation gives 0.443 to 0.349 us (1.271x).
The production AOT symbol used to remain on that FP32-absmax implementation;
after connecting this ordinary Triton graph to `_quantize_bf16_w8_kernel`, an
old/new bundle-object comparison is 0.500 to 0.349 us (1.433x). With the same
BN64 GEMV, the full K=N=1024 generated-symbol pipeline is 27.063 to 26.769 us
(1.011x). The rebuilt Qwen3 bundle audits all eight selected quantizer objects
for exactly three UMAX, two FMAXNM, four round-even conversions, and no FMINNM,
call, or stack reference. Its cache key also includes imported kernel sources,
so this connection cannot be hidden by a stale AOT object.
Before the native RNE and bounded K32 schedule were added, BLOCK_K=32, 64, and
128 generated about 1.7k, 4.8k, and 4.3k lines respectively with no latency
benefit. Large, bandwidth-bound shapes retain their explicit tile selection
and must not be included in a blanket shape-independent replacement.

## Direct-output change

The SMMLA lowering keeps the K loop rolled and writes each final 2x2
accumulator fragment directly to a strided output descriptor when the dot has a
unique, unmasked, in-bounds store and a zero initial accumulator.  It no longer
allocates a full int32 output tile and reads it back for the final store.

This reduced direct latency from about 26.76 us to 20.49 us for
128x128x128/BM=128 (about 23%) and from about 1.586 us to 1.52 us for
32x32x128.  For 256x256x256/BM=256, the generated assembly is 322 lines with
16 static `smmla` instructions; LLVM IR contains only the 2 KiB B-packing
buffer and no full output accumulator.

## Reproduction environment

```bash
export PYTHONPATH=python
export TRITON_BACKENDS_IN_TREE=1
export TRITON_CPU_BACKEND=1
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=$PWD/artifacts/a-fresh-cache
taskset -c 0 /home/kevin/venv-int8-clean/bin/python \
  -m pytest -q --device cpu \
  python/test/unit/language/test_core.py::test_unroll_attr \
  python/test/unit/cpu/test_target_info.py \
  python/test/unit/cpu/test_sve2_i8mm.py
```

The target-selection and SVE2/I8MM unit suites report 20 passing tests in
default mode and 19 passing plus one intentional SVE-only skip in forced-fixed
mode. They check
numerical correctness, target/VL detection, emitted SMMLA and SDOT, absence of
scalar multiply-long expansion, and absence of a full 32x32 int32 accumulator
buffer on the direct-output path.  It also checks that CPU honors
`loop_unroll_factor=2` for the packed W8A8 SDOT loop.  The Q4-prefill
regressions construct KAI blobs without FlagGems.  The Q4 case runs the
ordinary M16 Triton kernel, requires 64 SMMLA calls and no residual
`triton_cpu.dot`, and compares the BF16 result against an independent PyTorch
reconstruction.  The W8 case runs a K1024 loop, requires 64 SVE SMMLA calls,
zero folded spill/reload and no runtime symbol, then checks that a different
physical permutation is not rewritten.

The portable fixed-width production check is:

```bash
TRITON_TEST_PYTHON=python3 \
  bash benchmarks/run_q4_router_m4pro.sh
```

It forces fixed-I8MM, runs the Q4 production router across decode, M4/M8/M12
tails and M16 blocks, audits every active Q4/Q8 object, executes W8 decode and
prefill correctness, exhausts finite-BF16 quantizer inputs, and checks joined
QKV plus three-stage W8 SwiGLU before timing the production routes. The script
completed successfully on CIX in forced-fixed mode; that proves the portable
instruction path and runner composition, not Apple latency.

Packed W8A8 benchmark with the factor-two default:

```bash
PYTHONPATH=python:/home/kevin/triton-opt-cpu/benchmarks \
taskset -c 0 /home/kevin/venv-int8-clean/bin/python \
  /home/kevin/triton-opt-cpu/benchmarks/bench_w8a8_codegen.py \
  --k 1024 --n 3072 --grid whole --warmup 5 --iters 20 --batches 5
```

Exact-KAI-layout W8 compile audit and direct KleidiAI comparison:

```bash
export TRITON_CACHE_DIR=$PWD/artifacts/cache-kai-w8-repro
PYTHONPATH=ports/triton-cpu-3.7.2/python:third_party/FlagGems/src \
  /home/kevin/venv-int8-clean/bin/python -S \
  benchmarks/bench_w8_kleidiai_layout_codegen.py \
  --k 9728 --n 2560 --unroll 2 --mode auto --compile-only

export TRITON_CACHE_DIR_OLD=$PWD/artifacts/cache-kai-w8-flattened-repro
TRITON_CACHE_DIR=$TRITON_CACHE_DIR_OLD \
PYTHONPATH=ports/triton-cpu-3.7.2/python:third_party/FlagGems/src \
  /home/kevin/venv-int8-clean/bin/python -S \
  benchmarks/bench_w8_kleidiai_layout_codegen.py \
  --k 9728 --n 2560 --unroll 2 --mode pointer --flatten-k8 --compile-only

benchmarks/cpp_wrapper/build.sh
taskset -c 11 artifacts/bench_w8_kleidiai_layout \
  "$(find "$TRITON_CACHE_DIR" -name '_kai_w8_layout_pointer_kernel.so' \
       -type f | head -n 1)" \
  9728 2560 100 21 4 pointer \
  "$(find "$TRITON_CACHE_DIR_OLD" \
       -name '_kai_w8_layout_pointer_kernel.so' -type f | head -n 1)" pointer
```

Passing compile audit requires 16 SDOT, one ADDP, ten LD1R, zero lane inserts,
zero stack load/store, zero external call, and zero residual dot.

Q4_1 benchmark with launch overhead:

```bash
PYTHONPATH=python:/home/kevin/triton-opt-cpu/benchmarks \
taskset -c 0 /home/kevin/venv-int8-clean/bin/python \
  /home/kevin/triton-opt-cpu/benchmarks/bench_w4a8_q41_codegen.py \
  --k 9728 --n 2560 --warmup 5 --iters 20 --batches 5
```

Direct Q4_1 comparison after locating the generated kernel `.so`:

```bash
taskset -c 0 /home/kevin/triton-opt-cpu/artifacts/bench_w4a8_q41_microtile4 \
  KERNEL_SO \
  /home/kevin/triton-opt-cpu/artifacts/libw4a8_q41_microtile4_c.so \
  9728 2560 500 9
```

Direct production Q8 pack-plus-matrix after compiling the two kernels:

```bash
/home/kevin/triton-opt-cpu/benchmarks/cpp_wrapper/build.sh
taskset -c 11 \
  /home/kevin/triton-opt-cpu/artifacts/bench_q8_generated_pipeline_aot \
  PACK_SO MATRIX_SO 16 1024 1024 2000 21 300

cd /home/kevin/triton-opt-cpu/third_party/libtriton_jit
export TRITON_BACKENDS_IN_TREE=1
export PYTHONPATH=/home/kevin/triton-opt-cpu/ports/triton-cpu-3.7.2/python:/home/kevin/triton-opt-cpu/third_party/FlagGems/src:/home/cix/venv-fep-e2e/lib/python3.11/site-packages
taskset -c 11 build-cpu/tests/bench_cpu_q8_pipeline 16 1000 15
taskset -c 11 build-cpu/tests/bench_cpu_q4_pipeline 16 500 15
```

These results are local single-core measurements.  Multicore scheduling,
thermal stability across machines, and other Arm vector lengths still require
separate evaluation.
