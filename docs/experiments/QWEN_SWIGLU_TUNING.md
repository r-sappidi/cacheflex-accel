# Qwen FFN analytical tuning and prefill projection

Related ledger IDs: AI-03/04 already swept KC, core mapping and gate/up combination
on older synthetic workloads; first-call packing and repeated-call timings differed.
AI-06 established fully checked FP16-storage/FP32-accumulation model-shaped SwiGLU
on D2048/F8192. AI-08 identified large logical partial-accumulator traffic without
proving its additive latency contribution. AI-09 is a finite-model transfer of
that older workload, currently active, not Qwen tuning. L3-06/07 are the Qwen FA
transfer and analytical search. New entry AI-10 transfers the existing FFN guest
to Qwen D896/F4864 and the exact current Qwen FA simulator/profile, then tunes
cache and private SPM independently with equal bounded candidate budgets.

The decision-changing outcome is a lower complete SwiGLU first-invocation ROI
against each architecture's inherited MP2/KC128/separate configuration, followed
by strongest accepted cache versus strongest accepted SPM. Null and regressive
cases remain in the report. No new near-L3 FFN arithmetic engine is proposed.

## Workload and hardware

Qwen2.5-0.5B-Instruct's frozen model_config.json supplies hidden size896,
intermediate size4864, 24layers, 14query/2KV heads, d64. Batch1 synthetic tensors;
FP16 storage and FP32 sequential GEMM accumulation, fused SiLU-product rounding.
This is model-shaped arithmetic, not execution of a BF16 checkpoint.

The FFN is gate=SiLU(X Wgate), up=X Wup, Y=(gate * up) Wdown.
The two input projections are T x896 by896 x4864; down is T x4864 by4864 x896.
Every output is checked against the ordered reference and a separate float64
full-chain reference. Existing error bounds remain unchanged. Hash checks cover
guest/export consistency, prior compatibility controls, and exact equality
across mapping/KC/combination/backend variants with identical tensor shapes.

Use the unchanged lifecycle build_v5 simulator and t4096_cache configuration
from the Qwen FA lineage: eight2.5GHz cores, VL16, 64KiB L1/core, 512KiB private
L2/core, 4MiB shared LLC, finite queues/credits, repaired retry and common LLC
service port. Full config comparison permits only executable/argv/readfile and
SPM eligibility. SPM eligibility is six ways, matching FA; this guest uses four
ways, with actual panel claims of4 xKC x64bytes (32–256KiB). No extra data SRAM,
bandwidth, queues, or L3 reduction commands. Frozen old guests are copied
byte-for-byte; no simulator or kernel source change is needed.

## Selection and validation

scripts/qwen_swiglu_model.py enumerates MP1/2/4/8, KC128/256/512/1024, and separate
or combined gate/up. The existing eight-row,128-column FP32 kernel remains fixed.
MP partitions token rows;8/MP partitions output columns. Seven down-projection
column tiles make N partitioning uneven. Enumeration records per-core arithmetic,
SPM copies, packed-A bytes and partial-accumulator traffic, including KC tails
and empty row partitions. Capacity terms are explicit uncalibrated heuristics.
27 sensitivity scenarios per architecture select two robust candidates and one
different-MP diagnostic, plus the inherited baseline: four candidates each.
Scores are not predicted runtime, and the search does not prove global optimality.

1. Four old T24/D256/F512 controls on current hardware, exact old output bits.
2. All eight selected backend/configuration pairs at T24/D896/F4864, checking
   actual Qwen tails, combined paths and empty row groups. One simulator at a time.
3. Wait for the existing FA analytical coordinator to complete successfully.
4. All eight candidates at T1024. Independently select each backend's first-call
   winner. Reuse accepted controls instead of rerunning identical cases.
5. Transfer inherited control plus selected winner, if different, to T2048 and
   T4096. Retain the inherited target result as an eligible winner.

All runs include full numerical checks, output bits, exact expected copy counts,
zero reduction commands, zero global-guard rejects, and finite LSQ occupancy.
Any failed gate stops this queue; no tolerance relaxation or automatic restart.
One additional validation simulator now; at most two new simulators after FA.
Launch requires >=50% host CPU idle, >=128GiB available RAM, >=100GiB disk, and
fewer than six existing simulators. Runtime memory/disk floors64/75GiB. Host
reference uses one thread. Existing LLaMA-shaped SwiGLU and attention runs stay
untouched. State is results/ai_workloads/qwen_swiglu_20260928/experiment_v1/status.json.

## Timing and projection contract

Primary timing is the complete first invocation: weight packing/combination,
activation packing, SPM staging, all GEMMs, SiLU-product and synchronization.
Allocation, input loading and thread creation are outside the ROI. A second
invocation is reported separately as a reuse diagnostic. Two samples do not
establish steady warm behavior. In particular, do not add second-call FFN timing
to first-call FA timing and present it as a measured model execution.

The eventual projected prefill layer time is the SUM of matched QKV projection,
RoPE, FA, output projection, two RMSNorms, two residual additions, and SwiGLU
times. Projected model time adds24 such layers plus embedding/final RMSNorm and
the selected vocabulary-projection policy. QKV widths are896/128/128; output
projection is896 x896. Vocabulary width151936; last-position logits versus all
prefill logits must be specified before reporting a model estimate.

CacheFlex+L3 can initially reuse CacheFlex's measured FFN/component timing and
substitute only its FA timing. Do not invent an L3 FFN gain or average component
speedups. Remaining work after FFN is matched projection and small-operator
measurement, coherent intermediate-layout/copy accounting, a common first-use
versus prepacked deployment policy, and an integrated-layer check of the summed
estimate. Report component-composed projected prefill speedup with its inclusion
boundaries; no full-model, decode, or paper-readiness claim follows from FA+FFN
alone. Packing-inclusive first-use results and eventual prepacked projections
must be shown separately.
