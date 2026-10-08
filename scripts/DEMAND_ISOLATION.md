# Copy-induced ingress blocking: causal FA experiment

Hypothesis: a burst of SPM copies delays unrelated coherent L1 hits before
lookup, exposing stalls in the driver even though the V copies finish long
before their consumers. Measure the **full FA ROI latency difference**, not
summed copy waits or request counts.

## Intervention

`MessageBuffer.cacheflex_copy_demand_bypass`, default false, is supported only
on the unordered actual-L1 `l0_cntrl.mandatoryQueue`. It is mutually exclusive
with the older age-window `cacheflex_ingress_lookahead` mechanism.

After a plain SPMCP at the oldest queue position resource/protocol-stalls:

- Inspect existing admitted entries in age order, skipping copies as candidates.
- Select only ordinary cacheable LD/IFETCH, independent of every older entry
  and all requests in the protocol stall map. Same coherent source lines,
  stores, atomics, maintenance, uncacheable/strict accesses, unsupported SPM
  operations and transactions preserve the existing conservative boundaries.
- Selected requests execute the ordinary protocol and resource checks. There
  is no guessed-hit shortcut, forced completion, discarded copy, or free MSHR.
- At most four demand selections (successful or failed) per cycle, still
  consuming the original controller transition budget. The oldest request is
  attempted first on a fresh cycle. No policy change for an ordinary coherent
  head; thus the cache-only workload should be an exact no-op control.

No modeled queue entries, data SRAM, miss/writeback credits, ports, network
bandwidth, clock rates or fixed delays change. Requests stay in the original
heap; the selector's host vectors hold only references. **Searching that whole
heap has no separately modeled circuit latency/area.** This is a diagnostic
sensitivity experiment, not a realizable zero-cost CAM or an RTL paper result.
It is not a mathematical upper bound on all possible optimizations.

## Frozen platform and software

T512 gates, T4096 target; Hq=Hkv=1,d64,8cores,VL16. 64KiB L1I/D and private
L2/core,4MiB LLC, original retention and translated MICRO pools (L1I/D8misses,
L1write/bridge8,privateL2miss20/WB8,pooledLLCmiss32/WB16). Sequencer1024 guard.
Cache Br32/Bc128/G2; SPM Br16/Bc128/G4,V-only staging,two ways. Same guest ELF
bytes and ROI including preparation/synchronization. No additional tuning,
multicast, batch, DMA, dedicated SPM backing or storage increase.

## Queue and gates

Runner `scripts/run_demand_isolation.py`, isolated result root
`results/cacheflex_flash_attention/demand_isolation_20260922`; maximum two
new simulators, unrelated SwiGLU work untouched. Preserved pre-edit source and
previous frozen binaries remain available. New frozen build/config/input hashes.

1. 17 native dependency/eligibility assertions.
2. Enabled bridge roundtrip, no-barrier WB-RAW, cross-core release/GETX tests.
3. T512 OFF/ON x cache/SPM. Independently check every output, exact output hash,
   exact expected copy count, config differences, all-copy aggregate accounting.
   OFF and cache-ON must match every old finite non-host statistic exactly.
4. Bounded T4096 OFF/ON traces around the first panels. Find the same panel
   (first source 0x138040) and driver hit (0xa7980); quantify its actual queue
   wait. OFF must reproduce the previous trace-disabled bounded control.
   These early exits are diagnostic only, not FA numerical/performance results.
5. Full T4096 OFF/ON x cache/SPM with the same validation gates. Failure stops
   this experiment's peers/downstream cases, not unrelated projects.

The trace check establishes whether this policy actually alleviated the target
mechanism. If it did not, a null speedup cannot establish that the mechanism is
unimportant. A successful intervention with little whole-ROI improvement would
establish limited recoverable benefit from **this intervention at this point**.

## Reporting

- Primary: `(SPM_OFF - SPM_ON) / SPM_OFF`, and `SPM_OFF / SPM_ON`.
- Controls: same comparisons on cache; matched-platform cache/SPM ratios.
- Mechanism: identified driver-hit wait, panel readiness and consumer slack;
  bypass counts and copy timing are supporting, not end-to-end causal fractions.
- If positive, label the measured reduction as the benefit of the scheduling
  intervention, not an exact additive share of original runtime.
- Ruby ingress scheduling may over-couple resources relative to a physical
  pipeline. A positive result still requires a hardware/fidelity assessment
  before claiming a novel architectural contribution.

Authoritative live files: `experiment/status.json`, `REPORT.md`, `mechanism.json`.
