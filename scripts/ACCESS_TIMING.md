# Single-access CacheFlex timing

`CacheFlexTiming` is an opt-in debug flag; it adds observations only, not
latencies, queues, banks, or arbitration decisions. `--debug-start/end` bound
the trace. Optional environment variable `CACHEFLEX_TRACE_LINE=0x...` filters
message/flit events to one 64-byte physical line. Controller resource failures
and credit-pool events remain unfiltered for dependency context.

Recorded boundaries:

- CPU issue/address execution/commit/store eligibility/admission: existing
  IQ/IEW/Commit/LSQ/SPM_CPU trace flags, joined by instruction sequence number.
- Message enqueue, eligibility, dequeue, recycling and stalled-message wakeup.
- Exact failed generated SLICC resource check, with controller/address/resource.
- Miss/writeback credit allocation, release and failed availability checks.
- Network message-to-packet mapping, including original and cloned message IDs.
- Every flit's injection, each link's send/arrival eligibility, each router's
  arrival/pipeline eligibility/switch grant/crossbar traversal, and NI arrival
  versus protocol-buffer acceptance.
- Actual virtual-channel, credit, and ordering rejections encountered by the
  switch allocator; these events are evidence, not additive stall-cycle sums.
- Data installation: protocol transition executing the SPM DataBlk update.
  The later completion notification is not included in issue-to-install time.

No bank-arbitration penalty can be extracted because this protocol/configuration
does not currently reserve tag/data SRAM bank resources.

## Validation and use

`build_access_timing.py` freezes a separately linked simulator/config/source
snapshot. `run_access_timing.py` replays the accepted MICRO-resource T512 SPM
case, first with ProtocolTrace, then with the fine trace around a selected
long-latency LLC-hit copy. Both must match **all finite non-host statistics**,
complete exported output bytes, and config parameters (apart from output-local
redirects) against the independently numerically validated original. The
kernel, hardware parameters, and existing result files are not changed.

Build and experiment names may be set with `CACHEFLEX_TIMING_BUILD` and
`CACHEFLEX_TIMING_EXPERIMENT`. Defaults are `build` and `experiment`; existing
experiment directories are never overwritten. Results live below
`results/cacheflex_flash_attention/access_timing_20260922`.

`analyze_access_timing.py PATH_TO_EXPERIMENT` checks identity and timeline
continuity and writes `BREAKDOWN.md`, `breakdown.json`, and
`target_evidence.log`. All intervals are disjoint and sum to issue-to-install
elapsed time. For a multi-flit response, the additive path follows its tail;
other flits' overlapping activity is retained as evidence but not summed.
The JSON contains every packet/flit/hop event and detailed failed decisions.

## Interpretation

This is an elapsed-time partition of **one selected access**, not a kernel
bottleneck percentage or a prediction of removable end-to-end latency. Queue
waiting can reflect dependencies and downstream congestion. A resource's last
rejection identifies the wait until the next check; it does not imply that
removing that resource limit would independently save that entire interval.

The initial validation workload is T512/Hq1/Hkv1, eight cores, VL16,64KiB private
L2, with translated MICRO credits. It is **not** the historical T16K trace nor
the newer Hq32/Hkv8/512KiB-L2 scaling series. Do not mix their percentages.
