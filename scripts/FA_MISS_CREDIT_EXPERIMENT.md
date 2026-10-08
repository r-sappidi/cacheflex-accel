# FA: causal impact of actual L1D miss-credit pressure

Question: does the observed actual-L1D miss-credit admission bottleneck affect
end-to-end FA latency, and does relieving it expose a faster-SPMLD benefit?

## Fixed setup

Use the accepted IFETCH-on simulator/config/guest snapshots from
`ifetch_edge_20260923/build` and `experiment/runs/*_edge1`. Single-head FA,
T4096/d64, eight cores, VL16, 64KiB actual L1I/D and private L2 per core,
4MiB LLC. Cache and SPM retain their original tuned tiles/drivers and ROI
boundaries. SPM uses two existing ways for V; real SPMCP and LLC-retention
settings remain unchanged. No ideal staging or extra SPM backing is enabled.

Only `system.ruby.l0_cntrl[0..7].dataMissCredits.capacity` changes: **8,16,32,64**.
These are actual L1D miss credits, not sequencer entries. L1I misses, L1
writeback credits, private-L2 misses (20), LLC resources, CPU LQ/SQ/ROB, NoC,
timing, prefetching, and data capacities stay fixed. Increasing credit counts
is a diagnostic increase in permitted outstanding transactions, not a claim
that physical miss-tracking hardware can grow without area/power costs.

Three backends at every capacity:

- Cache control.
- SPM with normal direct-load latency.
- SPM with the existing faster direct-load-latency diagnostic (9→5 cycles).

The third backend measures interaction: if faster SPMLD only becomes useful
after relaxing coherent admission, that supports the masking hypothesis.

## Execution

Same frozen binary as all original controls; no rebuild or simulator edits.
Reuse the six accepted 8-credit T512/T4096 controls after rechecking independent
arithmetic, export and printed hashes, configuration, copy counts and global
guard counters. No redundant full control rerun. Then nine changed-capacity
T512 arithmetic/config gates, followed by nine full T4096 runs, at most two
simulators concurrently. Evaluate64 first for a strong finite intervention,
then16/32 to show dose response/saturation. T512 timings are gates, not the
target-workload performance conclusion.

Every new case must pass independent full-output arithmetic, exact reference
output hash and printed hash, unchanged source copy count, exact config audit
allowing only the eight declared capacity edits, and nonbinding global
sequencer guard. Frozen input manifests,3h/case and50GiB free-space guard;
technical failure stops this queue. Existing SwiGLU jobs are untouched.

## Interpretation

Primary measurements: first-invocation FA ROI, relative to the same backend's
8-credit control; cache/SPM speedup at matched capacity; fast-vs-normal SPMLD
runtime difference at each capacity. This is a sensitivity experiment, not an
additive latency attribution or a paper-ready hardware proposal.

Secondary measurements: per-core L1D fullChecks/maxLive/occupancy, lower-level
credit stalls, sequencer ingress rejects, and ROB-head PC counters. Failed
checks count attempts/retries, **not lost cycles**. If64 still fills, do not
claim admission is completely removed. If L1 stalls fall without runtime
improvement, the tested intervention lacks end-to-end benefit; downstream or
CPU limits may have become binding. Equal benefit in cache and SPM is a common
platform improvement, not CacheFlex-specific speedup.

Run: `python3 scripts/run_fa_miss_credits.py` (one coordinator only).
Results/status/live accepted-case report:
`results/cacheflex_flash_attention/l1d_miss_credit_20260923/`.
Five host tests exercise positive/negative config audits and argument handling.
