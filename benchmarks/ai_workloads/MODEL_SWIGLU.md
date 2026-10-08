# Model-shaped SwiGLU pilots

This replaces exhaustive refinement of T64/T128, D512/F1536 screening points.
Their accepted results remain intact; they are not model-scale conclusions.

## Target and scope

First target: Llama-3.2-1B-shaped SwiGLU, D2048/F8192, T128/512/2048.
[Meta's definition](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)
specifies dim2048, FFN multiplier1.5 and rounding to256. The standard Llama
SwiGLU width calculation gives8192. The three FP16 weight matrices total96MiB.
This is a dense operator benchmark with synthetic Gaussian inputs/weights,
not pretrained checkpoint inference, perplexity validation, or full-layer time.
T4096 and a second model shape are reserved for later held-out evaluation.

FP16 operands/intermediate tensors, FP32 GEMM accumulation, FP32 activation
arithmetic in custom kernels. This is a chosen precision deployment variant,
not a claim to reproduce a native BF16 checkpoint. The old FP16-accumulation
results cannot establish performance of this path. Eight-row FP32 microkernels
are existing precision-study contenders, not already-proven optimal assembly.

## Hardware and comparison

Frozen repaired Ruby simulator; eight cores, VL16,64KiB L1I/D per core,
512KiB private L2,4MiB aggregate LLC. SPM has four eligible L2 ways, dynamically
claimed; no additional backing, DMA, multicast, batch or retention changes.
Baseline and SPM each receive the same hardware, ROI and numerical requirements.

Initial custom mapping: separate gate/up,2M partitions,4N partitions,KC128.
This transfers a prior T128 starting configuration, not its optimality claim.
Both separate and combined FP32 variants receive small numerical gates before
promotion. Large pilot cases initially use the separate variant only.

ACL uses the pristine pinned library and the public experimental low-level
`CpuGemmAssemblyDispatch` API. Its FP16-input/FP32-accumulate GEMM, weight
packing, workspaces, scheduling, SiLU and multiplication remain library-owned.
The harness uses ACL memory helpers for first-call preparation and persistent
weights. This is explicitly not the earlier untouched NEGEMM API composition:
in this revision `CpuGemm::init_assembly_metadata` omits `use_fp32_acc`, whereas
the public low-level wrapper forwards it. Merely setting the flag on NEGEMM
would silently leave FP16 accumulation enabled. No library patch/kernel filter
is applied. `fast_math=false`, `use_fp32_acc=true` request the mixed path.
IMPORTANT: the pinned library's mixed-precision SVE kernel requires SVE2,
whereas this simulated ISA advertises SVE1 only. ACL must choose a supported
fallback, not that SVE2 kernel. Do not present its ratio as a comparison against
an optimized mixed-precision SVE production kernel. The custom SVE cache
contender is essential; baseline selection must use the fastest valid contender.

## Measurement and validation

Each pilot has five complete invocations: report first separately, omit second
from the warm summary, and report median/range of the last three. These are
pilot samples, not proof of steady state. Spread is descriptive and cannot
block progress. All GEMMs, SiLU/product, packing/staging and synchronization
are measured. Allocation, file I/O and numerical checks are outside ROI.
Output checks between calls touch the output in all backends; disclose this
cache-state effect. ACL now measures repeated calls too, not one warmed sample.

Full independent ordered-FP32 references, finite outputs, per-call hashes,
export-hash agreement, copy counts, and complete config audits are mandatory.
Custom activation uses an approximate SVE polynomial/reciprocal; its scalar
reference explicitly emulates that arithmetic. The first attempted reference
used mathematical exp instead, causing three near-zero pointwise mismatches
in both otherwise bit-identical custom guests. Repairing the oracle exactly
reproduced all6144 exported values without changing guest bytes or tolerances.
The independent float64 full-chain audit remains unchanged (relative L2<=1%,
peak-normalized error<=2%). The scalar activation is additionally checked
against mathematical SiLU-product to two half ULPs plus2e-7 absolute.
ACL stages are independently checked using its actual intermediate inputs;
its materialized half SiLU has different rounding from the custom fused helper.
Never require bit equality between these different activation conventions.
The ACL SVE SiLU oracle also emulates its actual arithmetic (FP32 exp followed
by FP16 add/reciprocal refinement/multiply). An ideal-SiLU two-ULP local check
incorrectly rejected two values in the small gate: ACL legitimately differs
by three ULPs there. Scalar arithmetic matches all12,288 values exactly; all
three GEMM stages also match exactly. Complete float64 relative L2 error is
0.0625%, below the unchanged1% full-operator limit. Local mathematical error
is reported, not substituted for arithmetic validation or full-chain checks.

## Execution and next decisions

`build_model_swiglu.py` builds isolated guests; `run_model_swiglu.py` records
hash-frozen inputs/configs/simulator/guests and runs at most two simulators.
Authoritative output: `results/ai_workloads/swiglu_model_pilot_v4_20260921`.
Five small gates precede custom cache/SPM/ACL pilots, ordered by increasing T.
Failures stop promotion; no automatic tolerance changes. All raw stats remain.
Host tests: `OPENBLAS_NUM_THREADS=1 python3 benchmarks/ai_workloads/test_model_swiglu.py`.

The ACL gate exposed a Ruby software-prefetch liveness bug: completed early-ack
prefetches freed sequencer entries without notifying retry ports. When they were
the last outstanding work, the blocked load/fetch ports never woke. A narrow
Sequencer.cc fix calls the existing retry notifier after prefetch completion,
with no capacity or latency change. V3 uses the same fixed simulator for every
backend; all four custom gates reproduce old outputs/counters exactly. ACL
now completes, and the activation-oracle repair above validates its outputs.
V4 reruns all five gates with the repaired oracle before large pilots.
See `results/ruby_swprefetch_retry_fix_20260921/DIAGNOSIS.md` for trace evidence.

After pilots, use runtime and traffic evidence to select a small equal-budget
KC/core-mapping/fusion refinement for cache and SPM independently. Do not
automatically launch another exhaustive Cartesian sweep. Select winners by
absolute latency per architecture, not by the largest ratio, and compare SPM
against the faster valid custom-cache/ACL contender at matched precision.
Validate finalists on extra seeds and held-out shapes; model quality and core
scaling still need separate evidence before a paper-ready broad claim.
