# Coherent-load critical-path diagnosis

This observation-only experiment follows the completed matched IFETCH edge
study. It uses the accepted **IFETCH-on cache and SPM** configurations, not the
fast-SPMLD variant. No kernel, timing, capacity, or arbitration change is made.

## Scope

- FA T4096, single head, d64, 8 cores, VL16.
- 64KiB actual L1I/D and private L2 per core; 4MiB shared LLC.
- Existing cache Br32/Bc128/G2 and SPM Br16/Bc128/G4, V-only/two ways.
- Core 0 and 7 dependency traces; selected-PC Ruby events from all cores.
- SPM PCs 0x4020c0/0x402120 (QK coherent K), 0x402a58 (O rescale).
- Cache counterparts 0x402040/0x4020a0 and 0x4029d8.
- Predeclared windows 2.890–2.914 and 3.800–3.824 billion ticks; diagnostic
  prefixes stop 400,000 ticks after each window. CPU cycle is 400 ticks.

## Observation chain

Existing CacheFlexDeps supplies dispatch, source register provenance, actual
wakeups, readiness, issue, full-load completion, and architectural commit.
New CacheFlexLoads supplies address generation, translated read entry, LSQ
ordering waits/forwarding, every fragment's attempt/admission/return, and Ruby
admission rejection reasons, enqueue identity, and completion source.
CacheFlexTiming and ProtocolTrace preserve controller/transport context.

The packet pointer + core + bounded lifetime joins CPU and sequencer events;
RubyRequest identity joins mandatory queue enqueue/dequeue. The analyzer never
joins transactions solely by an address. It verifies the complete fragment byte
count and identifies the final-returning fragment, including when it was issued
before other fragments. Normal retries using the same request/fragments are
included. Reallocated, uncommitted, boundary-truncated, forwarded, and
nonmonotonic loads are excluded and counted explicitly. Initially-ready loads
can bypass the explicit IQ readiness hook; their source-ready bits prove
readiness at dispatch. Missing readiness is otherwise recovered only from
recorded source wakeups, never assumed from an unknown producer.

Disjoint per-load intervals sum from dispatch to register-ready. The remaining
response interval starts at the **last fragment's acceptance**, so earlier
fragment service overlapping issuance is not double counted. Separate per-
fragment service distributions retain that overlap information. Queue ready to
dequeue includes controller/resource stalls; it is not a measurement of pure
bank arbitration. Address to translated read includes translation/retry time.

These are sampled **per-access lifetimes**, not additive percentages of FA
runtime and not proof that eliminating an interval would save that amount of
end-to-end time. Existing full-ROI PC stall counters provide context, and a
subsequent matched intervention would be needed for causal runtime attribution.

## Safety and validation

`build_coherent_load_trace.py` freezes a separate binary/config/source snapshot.
`run_coherent_load_trace.py` runs at most two simulators; it has one-hour/case,
1GiB compressed trace/case, and 50GiB free-space guards. Any failure stops its
peer and downstream cases. Existing SwiGLU runs and inputs are untouched.

Before tracing, both T512 backends must reproduce all finite non-host stats and
output hashes exactly, pass independent full arithmetic checks, config equality,
and nonbinding global-guard checks. Every T4096 traced prefix has a same-stop,
untraced repeat and must match all finite non-host stats before analysis.
No incomplete prefix is reported as a newly validated full-ROI performance run.

Results: `results/cacheflex_flash_attention/coherent_load_path_20260923/experiment_v4`.
v3 passed T512 and exact early traceON/OFF checks but its analyzer incorrectly
excluded normal load retries and initially-ready MemDepUnit insertions. v4
repairs only analysis and reuses those validated gates/traces after checking
their original inputs, hashes, configurations, and counter equality. Previous
analysis scripts are preserved; new per-load reports live under v4/analysis.
The v2 early traces completed, but the exact-stats gate caught debug logging
calling MicroResourcePool.full() twice. Only eight fullChecks counters per
backend differed; every other finite non-host counter matched. build_v2 reuses
the original short-circuit check result and repeats all validation in v3.
Prior traces/binary and the explicit counter diff are preserved in experiment_v2.
The original `experiment` preserves an argument-order launch failure; it was
fixed by removing inherited host debug options before constructing commands.
`status.json` is authoritative. Analysis automatically writes per-window JSON
and Markdown reports plus compressed per-load lifetime records.
