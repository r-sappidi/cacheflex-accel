# Fixed FA execution/on-chip-memory headroom screen

User authorized one bounded2×2 screen for both cache and SPM after reviewing
existing evidence. This experiment does not tune new kernels or implement a
publishable hardware proposal. It asks whether substantial end-to-end headroom
exists in execution, on-chip memory, or their interaction.

## Constraints checked before targets

The same frozen `ifetch_edge_20260923/build/gem5.opt` and accepted kernels are
used. Normal configuration is IFETCH-on,8cores/single-head/T4096/d64/VL16,
64KiB L1I/D and privateL2 per core,4MiB totalLLC, two eligible SPM ways. The
eight-credit original remains the reference;64-credit results are diagnostics,
not a replacement control. First-invocation ROI and software work stay fixed.

Source/config audit:

- `cpu/o3/limits.hh`: compiled MaxWidth=12. Execution profile respects it.
- `cpu/o3/fetch.cc`: fetch buffer must fit the64-byte cache block. Retain it;
  do not quietly change cache-line size to remove fetch-boundary stalls.
- `arch/arm/ArmCPU.py`: condition-code register count defaults to5×integer
  registers. The execution profile explicitly preserves this derived ratio.
- `mem/ruby/structures/CacheMemory.cc` and accepted configs: cache-array
  resourceStalls=False, so SRAM bank-conflict arbitration is NOT exercised.
  Changing bank counts here would not provide a valid bank-contention test.
- SLICC L0/L1/L2/Directory enqueue actions consume the request/response service
  delays. CPU LSQ and FUPool consume the port counts and op latency/throughput.
  These are actual parameters in active paths, not unused cache-array latency
  settings. Existing traces locate pressure in these paths.
- Baseline L1Dmiss8/privateL2miss20/pooledLLCmiss32; protocolTBEs256/controller.
  Global sequencer guard1024 remains fixed and must show zero rejects.
- DRAM bus utilization is recorded for every controller from the control ROI.
  It is low in those controls, but that does NOT exclude individual DRAM
  latency effects. DRAM is deliberately not idealized in this screen.
- Normal MessageBuffers have buffer_size0 (unbounded), not a finite hardware
  FIFO calibration. Per-line micro coalescing limit16 and controllerTBEs256
  remain. Their rejection/occupancy effects must be examined in interpretation.

Exact generated profiles and control diagnostics are saved in constraints.json
before starting new simulations. Execution and memory parameter sets are
disjoint; combined is their exact union. Cache/SPM receive identical profiles.

## The two factors

| Resource | Normal | Relaxed factor |
|---|---|---|
| Fetch/decode/rename/commit widths | 5 | Execution:12 |
| Dispatch/issue/writeback widths | 8 | Execution:12 |
| ROB / IQ | 128 /80 | Execution:512 /256 |
| Integer/FP/vector/predicate physical regs | 128/192/192/64 | Execution:512/512/512/256 |
| Condition-code regs | 640 | Execution:2560 (derived5×integer) |
| Fetch queue | 32 | Execution:128 |
| Active integer/FP/SIMD/predicate arithmetic FUs | Existing counts/latencies | Execution:16each;1-cycle pipelined operations |
| LQ / SQ (regular/SPM per existing model) | 32 /48 | Memory:128 /128 |
| Coherent and SPM load/store issue ports | Existing2/1 and1/1 | Memory:16each |
| Active memory FU/AGU group | 4 | Memory:16 |
| Coherent/SPM controller service delays | Existing calibrated values | Memory:positive1-cycle floors |
| L1I /L1D miss credits | 8 /8 | Memory:32 /128 |
| PrivateL2 /pooledLLC miss credits | 20 /32 | Memory:128 /512 |
| L1/L2/LLC write credits | 8 /8 /16 | Memory:64 /64 /128 |
| Controller transition budget | Existing32,LLC4 | Memory:128 |
| NoC link width | 128bits | Memory:1024bits |
| Data/control VC buffering | 4 /1 | Memory:16 /8 |

Clock rates, topology, router/link hop latencies, branch prediction, pipeline
interstage delays, data cache capacity/associativity/replacement, prefetcher,
data precision, arithmetic semantics, true dependencies, translation and
coherence rules are unchanged. No oracle staging, dedicated SPM backing,
functional memory bypass, or added data SRAM. Larger tracking resources and
wider execution/transport have hardware costs that this diagnostic does not
estimate. Memory factor is **optimistic on-chip service, not perfect memory**.

## Validation and bounded execution

Reuse four original T512/T4096 cache/SPM controls after full arithmetic,
configuration, hash/copy-count and global-guard verification with the same
frozen binary. Six T512 changed-profile cases must pass before six T4096
targets. Every new case checks all output values against the independent
reference, export/printed hashes, expected real copy counts, complete config
equality except explicitly listed edits, finite pool occupancy and zero global
guard rejects. Counters verify active FMA and network paths; record remaining
pool/retry/FU/LSQ/ROB/fetch/DRAM constraints rather than assume they vanish.

Max2 new simulators;3h/case and50GiB free-space guards. Any error stops this
queue and its peer. Existing SwiGLU jobs and inputs are untouched. No automatic
extension beyond this screen. T512 is a validation workload, not a performance
conclusion about T4096. Kernel snapshots are unchanged, so no recompilation.

Current runner: `scripts/run_fa_headroom.py`.
Current results: `results/cacheflex_flash_attention/headroom_2x2_v2_20260923`.
Original `headroom_2x2_20260923` preserved its configuration-audit failure:
the implicit numPhysCCRegs increase was initially absent from the allowlist.
It is now explicit; old scripts/logs are retained. This was not a simulator
crash, arithmetic failure, or a relaxed correctness criterion.

## Decision outputs

Report full ROI for all four cells, matched cache/SPM ratio per cell, memory
benefit with normal vs relaxed execution, execution benefit with normal vs
relaxed memory, and combined benefit. The interaction term is
`T_execution + T_memory - T_combined - T_control`; positive means extra joint
savings beyond adding the individual time savings, not an additive attribution
of original runtime. These bundles are hypothesis screens, not an exact
classification of all CPU/memory costs (the subsystems interact).

Large gains shared by cache/SPM identify a general platform opportunity.
CacheFlex-specific headroom needs a matched relative advantage or a mechanism
that can realize the benefit more cheaply. Small effects cannot establish a
universal impossibility if any explicitly retained constraint remains binding.
Review one completed screen before choosing at most one targeted follow-up.
