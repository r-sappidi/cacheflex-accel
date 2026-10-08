# Accumulator dependency sensitivity (2026-09-25)

Hypothesis: coherent FP32 partial-sum loads limit SwiGLU even after weight
staging helps. Existing frozen T2048 PCs show these sixteen load instructions
accounting for about 72% of SPM and 49% of cache summed per-core
`pcNotExecutedCycles` in the fourth completed invocation. These are NOT
fractions of causal runtime loss. Selection/disassembly/hashes and all epochs
are saved in `results/ai_workloads/swiglu_accumulator_oracle_20260925/pc_profile.json`.

## Intervention

`RubySequencer.cacheflex_early_read_pcs` defaults empty. For an eligible
ordinary read with a selected PC, after normal admission and resource checks:

1. Reject the shortcut if another local request to the same line is in flight.
2. Read current data through Ruby's functional coherence-aware lookup. On
   failure fall back to the ordinary path; never read raw backing memory.
3. Insert an independent owning packet into the NORMAL request table and issue
   its NORMAL coherence request, retaining its normal request credit.
4. Return the functional snapshot to the CPU on the next Ruby clock edge,
   through the normal response port and its scheduling/backpressure.
5. When the real response arrives, compare all returned bytes with the early
   snapshot, panic on mismatch, free the normal request credit via the existing
   callback machinery, and wake retries. Do not deliver a second CPU response.

The background packet has no CPU sender state/data pointer, so early LSQ
retirement cannot leave a dangling CPU payload. LSQ address ordering and CPU
issue limits still apply before admission. Same-line aliases and masked reads
fall back. Only source-audited private accumulator loads are selected: output
ownership is disjoint in M/N and boundaries are cache-line aligned.

This is deliberately an impossible early-value oracle, NOT hardware, extra
SPM capacity, eliminated copies, an asynchronous store mechanism or a claimed
speedup. Normal memory work is still performed. Its ordering/timing and cache
hits can naturally change after computation starts earlier; exact traffic
identity is not promised. It tests exposed **read-value dependency latency**,
not all potential benefits of eliminating accumulator stores/traffic. A small
effect cannot rule out bandwidth pressure or store-side costs. The realistic
implementation's speedup is not bounded monotonically by this oracle because
shared-resource interference and ordering can change.

## Evaluation

Isolated build from current source, source originals retained under `preserved`.
Original SwiGLU and single-core jobs/binaries/configs untouched. New build uses
the old pilot's sizes, clock parameters and admission guard16, NOT the FA
MICRO-timing/resource profile. An adapter supplies newly required controller
objects with capacity0 (disabled); default independent-bridge mode is OFF.
Current simulator includes accumulated correctness fixes, so fresh OFF controls
are required. Do not combine an old-model control with a new-model oracle.

Runner `scripts/run_accumulator_oracle.py`:

- Two simulators maximum. T24/D256/F512 cache/SPM OFF, then ON gates.
- Same frozen FP16-storage/FP32-accumulation guest ELFs, 2M x4N, KC128.
- Then T2048/D2048/F8192 OFF pair and ON pair, two invocations each.
- First invocation includes packing; second reuses prepacked cache weights.
  Compare like invocation indices, not a five-call warm median.
- Exact ordered reference and exported hash, independent float64 accuracy,
  issued/completed SPMCP counts, background/early response parity, and full
  same-model OFF/ON config comparison (only selected-PC parameter differs).
- Gates require positive oracle use; report coverage/fallback counts rather
  than assuming every accumulator read took the shortcut.
- Failure stops this suite only. No variance gate, tolerance change or failure
  suppression. Post-exit checks must pass before reporting performance.

Initial experiment failed before simulation due to missing new resource-object
bindings. V2 supplied disabled objects but its old-model config audit rejected
the new default shared bridge-port connection. V3 explicitly audits only those
known plumbing differences; every old numerical hardware parameter must match.
Failed evidence retained. V3 passed all four gates, including every early-value
comparison and exact exported reference. OFF first-call timings reproduce the
old pilot (cache79.5064us,SPM40.12us). New oracle counters proved to be lifetime
cumulative, because Ruby resets an explicit list of old statistics. V4 corrects
reporting by differencing consecutive dumps and checks parity on the final
lifetime totals. V3's just-started targets were stopped before completing an
ROI; their partials are retained. V4 reuses all four accepted gates (unchanged
simulator/guests) and repeats the targets. Current status is
`experiment_v4/status.json`; coordinator1415502 at launch.
