# Online SPMCP latency aggregates

`CacheFlexAggregate.{hh,cc}` is an opt-in, host-side observer. It does not add
modeled memory, queues, events, ports, or timing decisions. Enable with
`CACHEFLEX_AGGREGATE_PATH=/absolute/path/aggregate.json`. Optional
`CACHEFLEX_AGGREGATE_START`/`END` select copies by their Ruby issue tick. The
default is the entire simulation. This version targets ordinary O3 SPMCP, not
multicast, batch, or DMA descriptors.

The scaled FA control uses T4096, Hq=Hkv=1, d64, 8 cores, VL16, 64KiB L1I/D
and private L2, and 4MiB total LLC. The unchanged SPM driver uses Br16/Bc128/G4
and two SPM ways; the cache driver uses Br32/Bc128/G2. These are existing
fixed candidates, not newly tuned optima. Both use translated MICRO resource
bounds, LLC retention, and a previously validated nonbinding sequencer guard.

Only in-flight copies, message identities and packets retain temporary records.
At completion, their disjoint intervals are folded into sums and fixed log2
histograms, globally, per core, and per source path. Completed per-copy records
are discarded. JSON is atomically replaced every 4096 completions and on normal
process exit. A crash can leave a partial snapshot: never treat it as complete.

CPU timestamps are IQ first issue, latest address execution, regular SQ writeback
eligibility, and Ruby `issueRequest`. Ingress aliases/deferred issue, if any, are
included before the last anchor. Installation is the successful private-L2
source transition into X; acknowledgement afterward is separate. Pure SPM loads
and SPM slot claims are not counted as data copies.

Generated SLICC observations mark first attempt and successful transitions;
failed resource checks partition the attempted-forwarding interval. Queue wait
behind older requests remains separate: a rejection count is not a duration.
Transport uses packet identity and message clone ancestry, following only each
tail flit. Summing all flits would double-count overlapping activity. Network
subintervals refine the encompassing source-service interval, never add to it.
Source paths distinguish LLC resident hits, misses, owner forwarding, and
private-source release, where observed. There is no SRAM bank arbitration model
to measure; Garnet router contention is a different resource.

Percentages divide aggregate component ticks by aggregate issue-to-install ticks.
They describe average request lifetime, **not fractions of FA execution time** or
independently removable delays. A latency-weighted breakdown may emphasize tail
requests; use per-core/source groups and histograms alongside it. Log2 bins
provide percentile bounds, not exact quantiles.

`run_access_aggregate.py` freezes all inputs, validates T512 observer-off/on runs,
then runs T4096 cache/SPM. All outputs are independently numerically checked and
must match prior exports exactly. Every finite non-host statistic must equal its
control, as must configuration except outdir paths. Aggregation must reconcile
the exact SPMCP count, have no unfinished records, no same-priority overlapping
intervals, and additive sums and histograms. Any failure stops this new queue,
preserving diagnostics and leaving unrelated active runs untouched.

`check_access_aggregate_trace.py` waits for the T4096 queue, then replays T512
observing only the selected copy admitted at tick274267600. Its 711.25-cycle
lifetime and eleven component sums must reproduce the previous explicit
packet/flit trace. This is a cross-check of accounting, not another performance
candidate or a representative workload sample.
