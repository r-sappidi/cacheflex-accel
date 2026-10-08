# Ordered MLP defensibility study

User requested execution of the six stages in order. Preserve all earlier
implementations/results. Do not assume a positive CacheFlex speedup.

## 1. Matched invocation / reusable weights

`mlp_study.cpp`, `build_mlp_study.sh`, `sweep_mlp_study.py 1 OUT BUILD`.
Cache shares persistent packed B for gate/up/down across cores. First-run
parallel packing is inside first-call ROI; warm invocations reuse that data.
SPM keeps scratch buffers alive but restages each required panel normally.
Both have identical thread lifecycle/barriers and per-invocation output checks
outside ROI. Check data warms output cache lines between calls; this is common
to both, explicit, and not included in measured time. Original kernels intact.

T24 smoke, then T64/T128 D512/F1536; original 24x128 tile, KC256, 2Mx4N.
First invocation separately; two warmups followed by three stats-reset/dump
epochs. Report individual times, median, spread, correctness for every call.
If warmed spread exceeds 2%, run a longer convergence check before treating
the median as steady state. Architectural settings remain the clean FA/MLP
configuration, including 32 MiB aggregate LLC. No DMA/multicast/batch.

## 2. Independent software tuning

Keep ACL in the contender set, not as the sole baseline. Staged search:
KC128/256/512 x grids1x8/2x4/4x2/8x1 at T64/T128, starting with 24-row
kernel. Test local 8/16-row derivatives independently (micro tree unchanged).
Original 24-row kernel remains a candidate. New row variants need functional
smokes before timing. Rank each architecture independently; retest winners
with two warmups/three measured epochs. Search samples with one warmup/one
measurement are shortlist evidence, not final steady-state numbers.

## 3. Symmetric MLP-level optimizations

After stage 2, compare combined gate/up projection against separate calls,
and reusable/in-place intermediate layouts. Both already use the same fused
SiLU-times-up helper; avoid asymmetrically giving only SPM that fusion.
Combined projection must preserve arithmetic and include any necessary
packing/copying in the appropriate first-call/steady-state ROI.

## 4. Real model dimensions

Select documented SwiGLU model dimensions rather than relabeling a GELU FFN.
Include multiple prefill T and separate decode T=1. Include production ACL for
each, especially decode where padding a 24-row GEMM is not a competitive GEMV.
Use independently generated nonzero references and record exact model config
source/revision. Large-shape sensitivity is not equivalent to running a model.

## 5. Common accuracy

Per-KC host references validate implementation; additionally compare all
finalists to the same high-precision mathematical chain. Predeclare an
accuracy requirement before interpreting speedups. Test a production-appropriate
FP32-accumulate path on both sides; do not compare its timing to FP16 accumulation
without disclosing the numerical mismatch. More than one input seed/range is
needed before generalizing numerical robustness.

## 6. Hardware sensitivity

Run paired LLC4MiB-total and current32MiB-total cases; distinguish bank size
from aggregate size. VL4/VL16 and1/2/4/8 cores need native compatible kernels,
identical paired hardware fields and topology appropriate for core count.
Treat VL4 on a merely narrowed VL16 tile as sensitivity, not a tuned native
VL4 best-to-best result. Preserve the current simulator and all existing
runner defaults. End-to-end ROI includes compute, packing/staging, activation
and synchronization; report preparation/amortization separately.

Progress/results root: `results/ai_workloads/mlp_study_20260909`.
Completion of one stage is not completion of the remaining stages.
