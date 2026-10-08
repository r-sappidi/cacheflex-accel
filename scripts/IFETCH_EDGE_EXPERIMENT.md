# Matched instruction-fetch timing diagnostic

Question: does the observed instruction-supply cadence hide the benefit of
faster SPM reads, and what is its full-FA latency impact on each architecture?

The existing PV loop is 19 instructions/76 bytes. With a64-byte fetch buffer,
its observed supply pattern is5/5/5/1/0/3/0 instructions per cycle. Completed
Ruby IFETCH responses enter PacketQueue at curTick(), which schedules no sooner
than curTick()+1; the CPU then consumes the response at its next clock edge.

The new default-off RubyPort parameter `cacheflex_ifetch_same_tick_response`
delivers an already-completed IFETCH response synchronously only when its
response queue is empty and not waiting for retry. It never overtakes queued
packets. A rejected send is enqueued and waits for the existing retry callback.
O3 IcachePort always accepts responses; the fallback exists for robustness but
these O3 runs do not establish its coverage for other CPU models/full-system use.
Data loads/stores/SPMLD/SPMCP never use this path. Icache tag/data latency,
miss service, resource bounds, widths, branches, buffers, kernels, SPM capacity,
and all other traffic remain unchanged. No new data SRAM or staging oracle.

This is a timing-sensitivity intervention, **not** an assertion that zero extra
response ticks or a particular same-edge event ordering is calibrated silicon.
The original model and all original binaries/results remain intact. A measured
benefit is not automatically a novel CacheFlex hardware contribution.

## Matrix and gates

Six cases: cache, normal SPM, fast-SPMLD SPM, each crossed with IFETCH intervention
off/on. Existing fast-SPMLD case changes privateL2 SPM response5→1; measured load
latency9→5cycles. Cache uses its existing tuned driver; SPM variants share the
same SPM ELF. This is fixed-configuration, not a new best-to-best tuning search.

All cases use8core, single-head d64, VL16 FA;64KiB L1I/D/privateL2 per core,
4MiB shared LLC, existing2SPMways and calibrated Ruby timing. Real SPMCP remains.

1. Full T512 off cases must exactly reproduce **all finite non-host counters**,
   output bytes and independent arithmetic against their prior references.
2. All three T512 on cases must pass full output/reference/hash, configuration,
   exact copy-count, nonbinding global guard and positive IFETCH-bypass checks.
3. Replay both T512 SPM interventions with bounded Fetch/Decode/Rename/dependency
   tracing. Require all finite non-host counters equal to their lightweight
   anchor-scan gates, then verify actual fetch cadence is below7cycles. Debug flags on the
   reference scan are observational; it only records low-volume SPMLD anchors.
4. Run six full T4096 cases. All262144 outputs independently checked, exact
   original-export/printed-hash equivalence, full config audit, copy counts and
   global guards. Off cases again require exact old counter equality.

Counters: `ifetchEdgeEligible`, `ifetchEdgeDirect`, `ifetchEdgeQueued`. Direct plus
queued must equal eligible. Default mode counters must be zero. Eligible counts
cover completed IFETCH responses, not data requests or cache misses.

Primary outcomes: each backend's on/off ROI reduction; cache/SPM speedup with
the **same** front-end setting; fast-SPMLD benefit with the intervention off
versus on. The difference between the last two measures tests interaction.
No cross-platform numerator/denominator mixing or additive bottleneck percentages.

Build/run: `python3 scripts/build_ifetch_edge.py`, then
`python3 scripts/run_ifetch_edge.py`. Source/config/ELF manifests and source
snapshots under `results/cacheflex_flash_attention/ifetch_edge_20260923`.
`experiment/status.json` is authoritative; `experiment/REPORT.md` includes
accepted cases only. Maximum2new simulators,3hours/case,1GiB compressed trace
per case,50GiB free-space floor; failure stops this queue, not unrelated work.
