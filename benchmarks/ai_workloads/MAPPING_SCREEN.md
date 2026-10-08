# Mapping-aware initial workload screen

Started 2026-09-17 by user request. This is a bounded first-stage screen,
not a completed tuning campaign and not an end-to-end model measurement.

## Hardware and isolation

Both architectures use the frozen repaired Ruby v3 simulator. Eight cores,
VL16, paper-style O3 core, 64KiB L1I/L1D, 512KiB private L2 and eight512KiB
LLC banks (4MiB total), prefetch and matched LRU. SPM has four eligible ways,
dynamically claimed by scalar copies. No multicast, batch, DMA, LLC retention,
or separate backing. Transaction recording disabled; ordinary PC/message
counters retained. This is distinct from the ongoing 64KiB-private-L2 FA pair.

Two simulator slots for this screen, at most four together with the FA pair.
Original source files, earlier binaries and results remain unchanged.

## Implemented workload families

| Family | Shapes | Interpretation |
|---|---|---|
| Grouped projection | 8 x (M32,K512,N512); 8 x (M64,K1024,N1024) | Distinct inputs/weights; no routing, gating, or full expert MLP |
| Shared-weight batch | Same shapes, one common weight matrix | Independent input batches; repeated-weight reuse |
| Pointwise convolution | M96,K512,N512; M784,K1024,N1024 | Plain NHWC 1x1 convolution via dense projection; no bias/activation |
| SwiGLU | T64/128,D512,F1536 | Full gate/up, SiLU-times-up and down projection |

Projection candidates: KC128/256; grouped/batched ownership either all eight
cores cooperate with M partitions1/2/4 or one core owns each group. Pointwise
M partitions1/2/4/8. SwiGLU candidates: KC128/256, M partitions1/2/4/8,
separate versus concatenated gate/up projections on BOTH architectures.
All use existing pipelined24x128 GEMM arithmetic. Total80 configurations,
160 paired-architecture simulations, plus12 small correctness-gate simulations.
Breadth-first round order evaluates every family/shape before refinement.

`analytical_screen.json` records weight/storage footprints, useful FLOPs,
SPM panel capacity and expected staging bytes. It does not predict cache
residency, DRAM miss rates or elapsed time. No scaled-cache extrapolation.

## Fairness, validation, and reporting

Three timed invocations per run: first includes required weight preparation;
the next two reuse persistent coherent packed weights. SPM retains scratch
allocation but restages panels. Same input tensors reused. First-call is
post-initialization, NOT a cold-DRAM claim. Numerical validation touches output
between calls for both paths. Warm median/spread is shortlist evidence only;
two subsequent invocations do not establish steady-state convergence.

Independent deterministic nonzero references check every output. Export uses
timing loads into drained64B scratch; exported FNV hash must equal guest hash.
KC-specific FP16 accumulation reference tolerance2e-6+1%abs(ref), plus common
float64 mathematical relative-L2<=1% and peak-normalized error<=2% gates.
This characterizes FP16 accumulation and is not a production-precision claim.
All invocations must pass; copies issued/completed must equal analytic counts.
Instantiated configurations are audited against the frozen seed and paired.
Sources, simulator, guest binaries and reference inputs are hashed.

Cache and SPM winners are selected independently per shape, separately for
first/warm objectives. No ratio-maximizing optimizer. Failures stop the queue;
already running peer may finish, but no downstream candidate is launched.
Do not treat a crashed/unchecked measurement as performance evidence.

## Explicitly unresolved after this screen

- Routing and full MoE operators; general3x3 convolution.
- Persistent activation/output SPM schedules and other placement families.
- Microkernel row variants, alternate loop orders and production-library controls.
- Multiple seeds and held-out shapes, converged warm runs and finalist retuning.
- These dimensions are NOT secretly queued. A poor result is inconclusive if
  important mappings remain unexplored. Existing MLP large-shape numerical
  failures are not waived by this new screen.

## Entry point

`python run_mapping_screen.py NEW_OUTPUT_DIRECTORY`

Current output: `results/ai_workloads/mapping_screen_20260917`.
Check status.json and processes before launching. accepted.json is per-run;
results.json contains completed audited pairs; summary.json appears only
after the entire bounded screen finishes. No automatic resume or overwrite.
