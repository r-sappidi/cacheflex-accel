# Model-guided FA tuning

This is a bounded, architecture-informed **ranking search**, not a claim to
predict an exact critical path or prove a globally optimal kernel.

## Fixed experiment

Single-head noncausal FA, d64, eight cores, VL16. FP16 GEMMs and inputs, FP32
online state, FP16 output. Same frozen Ruby model as the accepted KV experiment:
64KiB private L1I/D, 512KiB private L2, 4MiB shared LLC, same fabric, clocks,
resource/timing settings, retention policy and nonbinding sequencer guard.
No hardware changes, added SPM SRAM, DMA or multicast. All packing, staging,
computation, reductions, synchronization and release are inside the ROI.
Allocation/initialization and safe export are outside. First invocation, not
a guarantee of cold caches. Both cache and SPM use the same ELF and Q64 packer.

## Search dimensions and limits

- Families: query/cache, KV/cache, query/SPM, persistent-V KV/SPM.
- Br: 16,32,64; Bc: 128,256,512,1024.
- Query rows per group: 64,128,192,256,384,512,768,1024, divisible by Br.
- T4096 screen, then independent T16384 selection and refinement.

This gives 288 legal T4096 candidates and 384 T16384 candidates. We do NOT
simulate them all. Bc384 is excluded by the present SPM PV address-increment
implementation (tiles must not cross a 1024-set way-pair boundary), not by
an intrinsic hardware limitation. Register microkernels remain fixed.

Impossible live SPM allocation (>six slots per set), invalid alignment and
redundant group sizes are rejected with explicit reasons. Coherent working
sets exceeding cache capacity are scored, NOT rejected. Lazy SPM allocation
subtracts actual occupied slots, not all six eligible ways or partially
occupied whole ways. Per-set minimum coherent associativity is recorded.

## Model

Exact counts: copy payloads, group/loop counts, live SPM bytes and per-set
occupancy. Approximate features: logical K/V traversal, packing, accumulator
traffic, reduction traffic/barriers and L1/L2 footprint pressure. Traversal
bytes are NOT LLC misses or DRAM bytes. Nonnegative relative-error-weighted
regression calibrates these features against measured end-to-end ROI latency.
Its coefficients are NOT causal additive latency percentages: overlap and
correlation prevent that interpretation. Historical old/new packing results
are mixed and this limitation is recorded; new observations use common packing.

Initial sanity check trains on T4096 and tests on historical T16384. Persist
its errors and ranking correlation before fitting all available observations
for selection. Save predictions before each new batch and compare them with
actual measured latency afterward. No target result is accepted from the model.

## Execution and fairness

1. Eight independent arithmetic/config/export/copy gates, including large Bc
   and grouped-query tails; same-mapping cache/SPM hashes must agree.
2. Two new T4096 candidates per family: predicted winner and capacity-boundary
   exploration point. Same simulation budget per family.
3. Refit; two candidates per family selected DIRECTLY for T16384, not merely
   the best T4096 configurations transferred blindly.
4. Refit; two additional T16384 cases per family from neighboring Br/Bc/query
   group values where available. Record model errors and family winners.

Up to 32 simulations including gates, at most TWO running at once. Existing
cache/llama queues remain untouched. Previously generated references may be
reused only with their complete SHA metadata; case keys include T/Bc/partition
count. Every output element is independently checked with unchanged tolerances;
all config fields must match except explicit guest/SPM eligibility fields.
Failure stops only this queue and preserves evidence.

Boundary winners require further range expansion before declaring convergence.
The final summary is best-observed within this search and must also be compared
with the concurrent cache tuning and precision-different llama baselines. A
bounded search alone does not establish paper-ready best-to-best performance.

Commands (run once; outputs are immutable):

    python3 benchmarks/cacheflex_flash_attention/model_tuning/build.py
    OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s benchmarks/cacheflex_flash_attention/model_tuning
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/cacheflex_flash_attention/model_tuning/run.py

Outputs: `results/cacheflex_flash_attention/model_tuning_20260926/experiment/`.
`status.json` is authoritative; `*_plan.json` contains pre-measurement choices,
`*_prediction_check.json` contains prospective model errors. Legal and rejected
candidate lists and frozen input hashes are preserved.
