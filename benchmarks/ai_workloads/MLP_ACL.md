# ACL-backed SwiGLU baseline

`mlp_acl.cpp` composes upstream `NEGEMM` (gate/up/down),
`NEActivationLayer` (SWISH with a=1, i.e. SiLU), and
`NEPixelWiseMultiplication`. ACL owns compute kernels, weight preparation,
workspaces, multicore mapping, and its eight-thread CPP scheduler. There is
no custom GEMM, activation, packer, thread grid, or fusion in this driver.
The custom baseline and SPM implementations are preserved.

## Reproduce

From the repository root, choose new absolute output directories:

```bash
bash benchmarks/ai_workloads/build_mlp_acl_upstream.sh /absolute/new/upstream
OPENBLAS_NUM_THREADS=1 python3 benchmarks/ai_workloads/run_mlp_acl.py \
  /absolute/new/runs /absolute/new/upstream/guest
```

The recommended upstream script exports pristine commit
`c245aa08b328fe7ef1d356f40b89ebc587524dae` and builds with logging=0,
asserts=0, debug=0, SVE, FP16/FP32, and CPP threading. It neither imports the
local kernel-filter modification nor changes the original ACL installation.

For an existing library, `build_mlp_acl.sh NEW_BUILD` accepts `ACL_ROOT` and
`ACL_LIB` overrides. The build
records the checkout commit, its diff, the archive hash, source hash, and ELF
hash. The default existing checkout is c245aa08b328fe7ef1d356f40b89ebc587524dae,
not the earlier full-stack W7 README's revision. It contains an optional
`ACL_GEMM_KERNEL_FILTER` extension; the runner does not pass this environment
variable and the guest rejects it. That preexisting archive also logs
scheduler messages in ROI, so its initial runs are diagnostic only. The
accepted path must use the separate logging-free upstream archive. The runner
rejects CORE/INFO logging inside ROI.

## Measurement boundaries

- `first`: configure/allocate/initialize tensors and set up the thread pool
  before ROI. All three GEMMs' first-run preparation/packing, all five operator
  calls and their scheduling/synchronization are measured.
- `warm`: run the complete operator chain once before ROI, then measure a
  second invocation on the same constant weights and inputs. This includes
  activation packing and normal invocation overhead, while allowing ACL's
  persistent packed weights. It is a warmed-cache measurement, not merely a
  cold-cache prepacked test or proof of convergence over many iterations.
- Both exclude validation and printing. There are no internal phase timers.
- No caller-imposed 24-row padding: ACL receives the actual token count.
- FP16 tensors, fast_math=true, FP16 accumulation requested, native VL16
  (2048 bits), eight threads. ACL may choose different reduction blocking
  and intermediate rounding than the custom kernel; bit-exact cross-backend
  output equality is not required.

The runner freezes the simulator used by the clean MLP pair and copies its
exact input/reference files. It audits every config.ini field against the
corresponding custom-cache run, permitting only executable/command, output
paths, and the separately hash-checked readfile path. In particular this
retains 32 MiB aggregate LLC (eight 4 MiB banks), LRU, and two SPM load ports;
SPM eligibility is zero. No hardware changes, DMA, batch or multicast.

## Validation and interpretation

The smoke is T24/D128/F256, followed by T64 and T128 with D512/F1536.
The guest retains diagnostics against the archived FP16-chain reference at
`2e-6 + 1% * abs(reference)` but does not claim PASS from that comparison.
The old pointwise test failed at T64 (0.59% relative L2 difference), motivating
independent checks of intermediate outputs rather than loosening its tolerance.

`validate_mlp_acl.py` reads post-ROI gate/up/SiLU/hidden/output dumps. It
requires exact agreement for each GEMM with independent sequential FP16
multiply-add emulation on that operator's actual inputs, exact FP16 elementwise
multiplication, and SiLU within two output-half ULPs of NumPy float64 exp.
The emulation evaluates each multiply-add in float64 before rounding to half.
It also reports full-chain errors against a float64 mathematical reference
for both ACL output and the archived custom-schedule reference.

Nonfinite values, stage-check errors, protocol failures, incomplete runs,
config mismatch, or any SPMCP activity reject a result. First/warm outputs
must be bit-identical. The queue stops after any failed size; results from
failed cases are not accepted as performance evidence. Per-run
`numerical_audit.json` is the authoritative numerical acceptance record.

The ACL composition materializes FP16 SiLU before multiplication, whereas the
custom fused SwiGLU helper rounds after the product. The reference is a
specific FP16 accumulation convention, not an exact-real-arithmetic oracle.
The reported max absolute error and relative L2 error make deviations visible.

Do not divide a warmed ACL latency by the existing first-invocation SPM
latency and label it a matched steady-state CacheFlex speedup. A warmed SPM
counterpart and an independently prepacked custom-cache baseline remain
necessary for a best-to-best steady-state study. An ACL-backed operator is
also not a complete transformer or model benchmark.
