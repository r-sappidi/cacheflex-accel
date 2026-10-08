# Non-intrusive FlashAttention attribution

## SPM capacity probe and qualified removals (2026-09-09)

`run_spm_capacity_probe.py NEW_OUT BUILT_SIMULATOR` runs native VL16
T128/T1024 cache, shared-capacity SPM, separate-backing SPM, and legacy-SPM
controls. Each of the first three receives an exact same-guest trace-on/off
check. Original guests remain preserved. The traced SPM production ELF is
stripped from its debug-side rebuild, giving exact .text identity; it is not
claimed byte-identical to the legacy guest. Legacy timing is reported separately.

`RubyCache.spm_separate_backing` defaults false. With it enabled, the eight
coherent ways retain the same set mapping/capacity, and four additional slots
per set back logical SPM ways 0–3. Physical ways 8–11 never accept coherent
data, even when SPM slots are released. This adds 256 KiB of backing to the
512 KiB private cache; ports, latency, banks and source coherence stay fixed.
It diagnoses the cost of losing coherent capacity, not area-matched speedup.

ProtocolTrace identifies each removal's immediate transition. RETIRED events
record architectural memory operations (excluding prefetch/SPM operations from
normal-buffer reuse classification). Any overlapping retired store before a
read makes old-content reuse ambiguous, rather than guessing predicate masks.
Issue/retirement matches use line/time, not dynamic instruction IDs; neither
these events nor overlapping service latencies are exact causal cycle savings.

`analyze_spm_residency.py` emits qualified_removals.csv/json, including unresolved
protocol events, write-before-read ambiguity, and reads without an observed
post-removal request. All T128 observer gates pass for cache and both SPM
capacity configurations; all removals have a protocol label. Separate backing
SPM is 37.7384 us versus shared 37.9452 us (0.55% reduction), with matching
outputs and 2048 completed copies. T1024 is still in progress.

Live state: `results/cacheflex_flash_attention/spm_capacity_probe_20260909/status.json`.
Its final REPORT.md is written only after both token counts pass all gates.

## Buffer-residency observer (2026-09-09; validation pending)

Opt-in debug flag `CacheFlexResidency` records physical page mappings,
cache tag allocations/removals, permission changes, replacement candidates,
SPM installs/releases/migrations, sequencer requests/responses, work boundaries,
and ROB-head PCs (when the existing PC profiler is enabled). It inserts no
guest instructions. A replacement probe is NOT counted as a removal until
the entry is actually deallocated. Permission distinguishes accessible data
from allocated transient tags, but not clean-exclusive versus dirty-modified.

`analyze_residency.py RUN DEBUG_ELF T` currently supports the preserved native
VL16 **cache baseline** and its printed FLASH_SHARED/FLASH_ADDR ranges. It
produces per-removal `evictions.csv`, resolved PC stacks, time-weighted buffer
occupancy, later read reuse and request-service diagnostics. Unresolved/shared
PCs stay explicit. The trace must start before initialization to capture page
mappings and initial tags. It uses 4 KiB SE pages and 64-byte cache lines.
Requests coalesced without a separate ISSUE record are excluded from request
latency pairing. Later reads show reuse, not necessarily a stall: prefetch or
an earlier refill may already have recovered the data. Neither overlapping
service times nor ROB-head phase attribution are additive causal penalties.

`run_residency_study.py NEW_OUT SIM DEBUG_ELF` gates T128/T1024 baseline
trace-on/off controls on identical output, extracted traffic/core counters,
timing and hardware configuration. It then measures T1024 with private L2
512 KiB/8 ways, 1 MiB/16 ways and 2 MiB/32 ways, maintaining set count and
modeled latency. These are diagnostic capacity/associativity ablations,
not physically costed hardware proposals or exact per-buffer cycle savings.

The baseline debug-side ELF has matching production `.text`. The SPM debug
build failed that identity check, so SPM phase/residency comparison is not
accepted or queued by this baseline study. Original simulator/guest artifacts
are preserved. Initial rebuild attempts compiled objects but did not update
the executable; their MLP regression failures do not test the protocol repair.
The queue explicitly checks the new debug flag in the linked executable.

Current queue/artifacts: `results/cacheflex_flash_attention/residency_20260909_linkfix`.
Check its `status.json`; no measurements are accepted until the above gates pass.
After successful baseline FA analysis, the queue resumes the repaired-simulator
MLP regression and ordered study in a separate subdirectory.

`run_8core.sh` accepts `CACHEFLEX_FLASH_PROFILE=1`. This enables observation
only: it does not insert guest instructions, change scheduling, or alter the
coherence protocol. `CACHEFLEX_GEM5_BIN` selects a preserved simulator.

## Network counters

`system.ruby.network.message_{packets,flits,bytes,received,network_ticks,queue_ticks}`
are indexed by `source__destination__vnet__req/resp_type`. Names are Ruby
controller names, so L1Cache means architectural private L2 and L2Cache means
architectural LLC in MESI_Three_Level_SPM.

- Injection is counted after VC allocation succeeds, once per destination;
  retries are not counted. Garnet expands multicast into destination packets.
- Source means the actual injecting controller, not a forwarded request's
  original Requestor field.
- Bytes are logical message size, including the modeled control header, not
  rounded flit bytes or data-payload bytes. Flits are injection flits, not
  hop-weighted link activity.
- Received packets and latency sums are counted on tail-flit acceptance.
  Latencies use gem5 ticks and match Garnet's tail-packet definition. Divide
  each sum by that route's received count, not its injected count.
- These are Garnet traffic counters. Direct architectural L1/private-L2
  CoherenceMsg buffers and memory-controller transactions do not traverse
  this network and are excluded. These counters do not identify data buffers.
- Reset/dump uses normal gem5 statistics epochs. In-flight packets can cross
  an ROI boundary; injected and received totals need not be equal.

## CPU/phase counters

`system.cpuN.commit` exports sparse PC histograms:

- `pcHeadCycles`: one sample at the start of each observed commit tick with
  a nonempty ROB.
- `pcNoRetireCycles`: the subset with no successfully committed microop.
- `pcNotExecutedCycles`: the subset of those whose sampled head is still
  unexecuted after the commit attempt. This includes operand and execution
  waits; it is not a count of cache misses.
- `pcRetiredMicroops`: committed microops by PC, not architectural instructions.
- `pcEmptyRobCycles`, `pcObservedCycles`: cover empty ROBs and total observed
  commit ticks. CPU sleep is not sampled.
- `pcSleepHeadCycles`: elapsed unticked cycles attributed to the head PC
  captured when an active context's pipeline idled. This captures long waits
  without scheduling extra simulator ticks.
- `pcSleepEmptyCycles`, `pcSuspendedCycles`, `pcSleepCycles`: distinguish empty
  ROB idle time from software-context suspension, and count all unticked time.
  Dump/reset callbacks clip these intervals at statistics boundaries.

Head cycles plus empty-ROB cycles equal observed cycles. No-retire and
not-executed are nested subsets and must not be added to head-cycle totals.
Head cycles plus sleep-head cycles form the phase attribution. Empty ROB,
sleep-empty, and software-suspended time remain explicit. Total observed and
sleep time should cover eight-core ROI time within clock-boundary rounding.
The profiler currently requires single-thread-per-core configurations.
Sparse histogram text keys retain full double precision to avoid rounding
distinct instruction addresses into the same printed key.

`build_profile_symbols.sh <new-directory>` builds debug side ELFs and verifies
their `.text` against production ELFs. Simulation executes the originals.
`analyze_attribution.py <result> <debug-elf>` resolves full inline DWARF stacks,
annotates instructions, and groups PCs by FA phase. Shared Q/P packing and
libc helpers remain separate where static PCs cannot identify the caller.
The source-line classification is specific to this source revision; regenerate
and review it if the kernels change. Each PC and its source stack is retained
in the JSON, so the classification is auditable.

This is additive **ROB-head attribution**, not exact algorithm-phase wall time
or causal stall savings. Out-of-order phases can overlap, and an instruction
at the head can wait for unrelated resource pressure. Empty ROB and inactive
time are explicit, not silently charged to compute. The report's ROI denominator
assumes the matched eight-core, 2.5 GHz paper configuration (400 ticks/cycle).

## Reproduction

After rebuilding gem5, run `run_attribution.sh <new-absolute-directory>`.
It freezes gem5 and runs profiling-off/on controls at T=128 and T=1024,
plus the reduced-footprint group4 diagnostic at T=1024. All use eight cores,
VL4, Br64/Bc96, scalar K-only SPM, and unchanged production binaries.

Accept results only after normal completion, finite matching hashes, complete
SPM copies, exact packet/flit reconciliation, and unchanged runtime/traffic
between profiling-off/on controls. Hash equivalence is not independent
high-precision numerical validation.

## Clean timer/policy comparison

The legacy cache kernel calls the clock even when its timing output is null.
`build_clean_baseline.sh <new-absolute-build-directory>` opts into
`CACHEFLEX_FLASH_SKIP_UNUSED_TIMERS` in the micro reference source, suppressing
only unused clock reads. Actual requested timings and default legacy builds
remain supported. It verifies production/debug-side `.text` identity.

Run `run_clean_comparison.sh <new-absolute-result-directory> <clean-ELF>`.
It freezes all guest ELFs and gem5, runs cache/SPM observer-off/on pairs at
T128/T1024 with private-L2 LRU in both, and adds group4 plus timer/policy
controls at T1024. No multicast, batch or DMA kernel is used. The opt-in runner
flag is `CACHEFLEX_FLASH_MATCHED_LRU=1`; defaults remain unchanged.

Validate with `validate_clean_comparison.py <result-directory>`. It checks
full hardware configuration matching, observer invariance, finite matching
hashes, and completed copies. For the clean debug ELF use
`analyze_attribution.py <run-directory> <clean-debug-ELF> --cache-line-shift 4`
to compensate for the opt-in source block; older debug ELFs need no shift.

Accepted results and factorial controls are documented in
`results/cacheflex_flash_attention/clean_comparison_20260908/REPORT.md`.

## Native VL16 comparison

Do not run the VL4 production ELFs with a wider architectural VL: their assembly
contains full-vector accesses against fixed 96/64-column layouts.
`build_vl16_clean.sh <new-absolute-build-directory>` separately builds a clean
cache and scalar K-only overlap SPM pair. These use the micro reference's
end-to-end 8x128 QK kernels and predicated 8x64 PV kernel. The baseline's unused
timers are disabled. SPM stages four 64-byte lines per K row into ways 0--3.
The old VL4 implementations and binaries remain available.

`run_vl16_clean.sh <new-absolute-result-directory> <VL16-build-directory>` freezes
the simulator/ELFs and runs T128/T1024 observer-off/on pairs, Br64/Bc128, eight
cores, VL16 (2048 bits), matched LRU. Other core settings remain as in the clean
VL4 comparison, including two SPM load ports. It is not the historical
single-core paper GEMM configuration that used one SPM load port.

`validate_vl16_clean.py <result-directory>` checks configurations, finite matching
hashes, completed copies and observer invariance. This port is tested only for
these token counts and scalar K-only staging; multicast/batch/DMA/V staging are
not supported by this VL16 harness. A/softmax/packing functions use their native
SVE loops; the GEMM tile width changes from 96 to 128, so cross-VL results are
not a pure width-only hardware ablation. VL4 source-line phase mappings must
not be applied to these different QK/PV implementations without updating them.
