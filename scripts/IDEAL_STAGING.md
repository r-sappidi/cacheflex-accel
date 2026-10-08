# Oracle staging diagnostic

Question: can removing repeated staging delay **and its interference** expose
an end-to-end FA advantage while keeping the same physical storage budget?

This is an optimistic software-model intervention, **not hardware**, a DMA
proposal, or a strict global performance bound. Ordinary instruction issue,
commit, translation, existing dependency checks and CPU ports are unchanged.

## Mechanism

Default-off `RubySequencer.cacheflex_ideal_staging_cache/controller` references
the existing private L2. For plain aligned 64-byte SPMCP into an already claimed,
stable X slot, a Ruby functional read obtains coherent source bytes. The oracle
writes the actual slot, then acknowledges on the next sequencer clock edge.
No copy request, response, source miss/writeback credit or network packet is
generated. No second data array, persistent operand mirror, or cache capacity
is added. Source cache residency/directory state is left unchanged, unlike the
normal move-out path; this is an explicit optimistic change.

First-time slot claims and dirty victim evictions still execute the normal
protocol. So do unsupported DMA/batch/multicast operations, absent/busy slots,
outstanding conflicting source/destination transactions, and failed functional
reads. A failed functional read never falls back to potentially stale DRAM
bytes. Each category is counted (attempt counts, not exclusive copy counts).
Issued/completed copy counters include oracle copies; real-path copy count is
total minus oracle copies. A one-cycle live source/slot interlock protects the
normal completion response, and normal CPU SPM shadow dependencies remain.

Scope: SE FA read-only operands after initialization, plus targeted ordering
regressions. Functional reads use Ruby's current coherence-visible data; this
does not establish support for data races, full-system context switches or
arbitrary concurrent source writers. Normal-path timestamp aggregation is
rejected with the oracle enabled rather than silently producing false results.

## Evaluation

`python3 scripts/build_ideal_staging.py` freezes a new executable/config/source
manifest. `python3 scripts/run_ideal_staging.py` runs at most two new simulators:

1. T512 cache/SPM default controls: all old finite non-host ROI stats exact;
   newly added oracle counters must be zero.
2. Oracle roundtrip, no-post-WB-barrier RAW, and release/GETX regressions.
3. T512 oracle and oracle+SPM-response-floor1, full independent arithmetic,
   exact old export, guest/export hash, copy accounting, configuration audit.
   Cache-only oracle wiring must be an exact no-op.
4. T4096 new-build controls and the same two oracle variants, with all gates.

Fixed original guests: cache Br32/Bc128/G2, SPM Br16/Bc128/G4, V-only two ways;
single head d64/VL16, 8 cores, 64KiB private L2, 4MiB shared LLC. MICRO-calibrated
timing, original prefetcher/network/resource policy. No tuning or ROI changes.
The local floor reduces service latency, not all CPU/adapter/port overhead.
Report primary ROI time and ratios vs BOTH current SPM and cache. Report oracle
coverage and remaining real copies before interpreting a negative result.

Prior simulator/results and active SwiGLU inputs are untouched. Results live in
`results/cacheflex_flash_attention/ideal_staging_20260923/experiment_v3`.

Earlier retained attempts: experiment v1 rejected the config JSON null encoding
before acceptance (None, not the INI string "Null"); v2 default controls were
exact but its ordering gate correctly rejected an unexercised oracle because
SPMCP's CPU packet is the eight-byte slot operand, not the 64-byte line payload.
Build_v2 corrects that eligibility check and repeats every gate.
