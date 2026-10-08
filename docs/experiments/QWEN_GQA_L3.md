# Qwen-shaped causal GQA transfer, 2026-09-28

User authorized starting immediately with a real model configuration.
Qwen2.5-0.5B-Instruct: Hq14/Hkv2, hidden size896, d64, batch1.
Source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json
This is an attention-kernel workload with deterministic distinct inputs, causal
masking and seven query heads per KV head. It is not full-model inference:
no weights/projections/RoPE are timed, and existing FP16 GEMMs/FP32 online state
are retained rather than the model's published BF16 storage.

Related FA-06/07 tested old Hq8/Hkv1 tiling/PV implementations; FA-08/09 found
near parity at long context despite traffic reductions. L3-05 validated the
direct register epilogue and bounded single-head tuning. New here: transfer
to Hq14/Hkv2 causal attention on the current finite model and L3 engine, with
head/query-group task scheduling. This is not a claim that multihead FA itself
is new. Existing frozen sources, accepted controls and unrelated runs stay intact.

All arms share one new ELF, deterministic head-major query-group tasks assigned
round-robin to eight cores after rotating query-group order between heads to
spread triangular causal work, and cooperative physical-KV-head packing inside ROI.
Each core owns a distinct L3 output bank; rows are drained/released before reuse
across tasks/heads. Future KV blocks beyond a query group are skipped; within
the boundary block probabilities for future positions are explicitly zero.
Each core retains a rolling private V tile in SPM, with no new multicast or
cross-head pinning. No simulator, storage, queue, or bandwidth changes.

Sequence: one-head unmasked output-regression gates; small causal Hq14/Hkv2
cache/private/L3 gates and depth-one stress; T1024 full checks; then matched
T2048 and T4096 cache/private/L3. Br16/R256, inherited cache Bc512 and
private/L3 Bc256, paired direct epilogues. No cache retuning or new broad sweep.
The inherited cache is a strong historical reference, not a measured optimum
for the new causal multihead workload.

Acceptance requires every output against independent sequential-FP16-GEMM,
FP32-online-state, mathematical-exp reference; small cases additionally dense
float64 per-head validation and exact causal first-row identity. Same-Bc arms
must be bit-identical. Exact commands, initializations, copies, wire bytes,
task ownership, model config deltas, LSQ/source/sink/bank bounds are checked.
No tolerance changes. A failed gate stops peers/downstream, preserving evidence.

Decision: lower full attention ROI versus cache and, separately, private-SPM,
including K/Q/probability packing, staging, causal masking, normalization and
drain/release. Report negative/null outcomes, per-core workload imbalance and
network/memory counters without interpreting stall sums as removable runtime.

At most two new simulations, nice10, alongside existing jobs. Launch gates:
>=50% CPU idle, >=128GiB memory available, >=100GiB disk, fewer than six live
simulators. During execution retain >=64GiB memory and >=75GiB disk guards.
Single compiler and single-threaded numerical oracle. Status/REPORT under
results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2 are authoritative.
Experiment_v1 was stopped before targets to add the causal-work rotation;
its partial evidence is preserved. All gates repeat on guest_v4/experiment_v2.
