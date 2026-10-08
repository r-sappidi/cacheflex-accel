# Finite-resource CacheFlex model

This is an **opt-in candidate hardware model**, not a calibrated production CPU
or a completed validation result. Existing frozen experiments are unchanged.
Its purpose is to make request capacity follow identifiable hardware structures
rather than a shared global Ruby sequencer limit.

## Resource contract

| Resource | Candidate bound | Lifetime / full behavior |
|---|---:|---|
| Regular load tracking | 32 instructions/core | Includes accepted loads surviving squash or early retirement; free on last response |
| Regular store queue | 48 instructions/core | Existing store ordering; bridge releases this entry on accepted send |
| SPM load tracking | 32 instructions/core | Same retained-credit rule, independent of regular LQ |
| SPM store/dependency tracking | 48 entries/core total | Pure SPM stores and outstanding bridge shadows share the budget; bridge shadow persists to completion |
| L1 controller TBEs (combined I/D controller) | 16/core | Allocation stalls when full |
| Private L2/SPM controller TBEs | 32/core | Coherent and bridge transient transactions share finite capacity |
| LLC TBEs | 64/bank | Includes coherence transients, not just DRAM misses |
| Directory TBEs | 64/directory | Explicit allocation resource checks, including SPM requests |
| Ruby controller queues | Usually16 messages | Private L1→L2 request/recall-return budget partitioned12+4; other buffers16; prefetch hints drop when full |
| Same-line coalescing records | 16/line | Refuse additional targets until an existing request completes |
| Garnet buffers | Existing 4 VCs/vnet; 1 control or 4 data flits/VC | Unchanged credit backpressure |
| Sequencer global guard | 1,024, compared with 4,096 | Diagnostic safety guard only; any rejection fails acceptance |

The 32/48 LSQ capacities are grounded in the supplied CacheFlex paper and the
current core configuration. The controller sizes are explicit, plausible design
choices requiring sensitivity testing, **not inferred silicon specifications**.
The MICRO tree's classic defaults (8 L1 MSHRs, 20 L2 MSHRs, 32 L3 MSHRs) provide
scale context, but do not imply equal Ruby TBE counts: TBEs also track protocol
transients, recalls and replacements. Sixteen per-line targets is consistent
with its L1 target default, not proof of an exact hardware match.

No extra SPM backing, extra cache capacity, added network bandwidth, DMA, batch
copy or multicast is introduced. Existing cache/SPM geometry and guests remain
unchanged. Pure SPM accesses remain independent of coherent LSQ capacity.
For this initial conservative bound SPMWB shadows share the SPM SQ budget too;
this is **not** an exact implementation of the paper's SPMWB placeholder-LQ
allocation. The target FA path uses SPMCP and SPMLD.

## Implementation details

`cacheFlexBoundedTracking` charges the existing shadow map against SPM SQ
capacity at dispatch. Completion releases the reservation and wakes dispatch.
The hash map is only the simulator representation of a finite table, not a
hardware hash table. Successful load fragments are counted until their replies;
squashed loads cannot free physical tracking and leave unlimited live requests.

`cacheflex_finite_ingress` checks actual mandatory-queue slots before admission.
Alias replays retain their existing tracked request and wait for an ingress
slot. The retry list contains references, not extra accepted operations or data.
This logic lives in the Ruby interface, but models **controller queue space**,
not a separate global outstanding-request pool. Same-line target accounting
likewise uses Ruby's existing coalescing representation.

Directory TBE allocations now use SLICC resource checks. Prefetch generation
checks queue space because its direct function enqueue is not protected by
transition resource checks. With legacy infinite buffers these tests do not
limit admission. Default-equivalence gates still verify the actual result.

### Finite-buffer progress repair

Initial T512 SPM validation deadlocked; both default-off controls reproduced
their accepted results exactly. A trace at ticks 256000000–260000000 establishes
an actual queue dependency: private L2 controller5 issues `L0_Invalidate_Own`
for `0x5e84c0` at tick257040000 and waits in `M_IL0`. L1 controller5 receives
`INV_OWN`, but cannot enqueue the dirty response: `bufferToL1` has16 occupied
slots (all in its stalled-message map, heap size0). The requests holding those
slots need the recall to finish. A diagnostic with queue limits removed, but
all other new bounds retained, completes.

The first repair reserved **one of the existing16 slots**, not an extra buffer, for
recall responses. Demand/prefetch/SPM and unsolicited eviction sends require
two free slots before consuming one; recall data/acks require only one. SLICC's
generated transition checks enforce this atomically before any action executes.
Legacy unlimited queues are unaffected. This is a progress requirement of the
finite-buffer profile, not evidence of a performance gain or exhaustive proof
that all possible protocol cycles are eliminated.

This v2 profile passed T512 cache/SPM correctness and all-counter invariance at
global guards1024/4096. However, the four-message stress case exposed a second
dependency: an already-enqueued recall response can sit behind a demand whose
processing needs space in the opposite-direction queue. That opposite queue is
itself blocked on another recall response. Reserving capacity alone does not
resolve FIFO head-of-line ordering.

The v3 repair gives recall data/acks a separate ordering domain, **partitioning
the same16-entry budget into12 request and4 response entries** (stress:2+2).
The existing, otherwise unused `spmBufferFromL0` input carries recall responses;
ordinary SPM requests remain on their original path. New return events duplicate
the original data/ack transitions and consume the correct input queue. The
default route aliases the original shared buffer and preserves legacy behavior.
`bounded_se.py` configures this connection using only the frozen configuration
copy; root configs used by active jobs are untouched. New builds require this
adapter, rather than invoking the unmodified `se.py` directly.

V3 passed all eight T512 gates but both T4096 runs aborted during initialization:
private L2 received `WriteBack` in `M_I`. The preserved cache replay at
`diagnosis/writeback_v3` establishes the race on `0x792080`:

- tick1297996000: L0 evicts dirty data, sending PUTX on the request FIFO.
- tick1297998500: private L2 starts an L0 recall (`M -> M_IL0`).
- tick1297999600: L0, now invalid, sends INV_ACK on the separate return FIFO.
- tick1298000500: that ack overtakes PUTX (`M_IL0 -> MM`).
- tick1298001000: private L2 starts eviction using its stale array (`MM -> M_I`).
- tick1298009500: delayed PUTX arrives in M_I and triggers the panic.

The v4 repair sends PUTX on the **same ordered return FIFO as INV_DATA/INV_ACK**.
The receiver applies the original writeback transitions using the new queue's
payload and dequeue actions. This preserves data-before-ack ordering, rather
than discarding the late dirty payload or merely suppressing the panic. The
same12+4 total budget, latencies, and legacy shared-FIFO alias remain unchanged.
V4's first bounded SPM gate exposed a second ordering bug: on `0x601f80`, L0
evicted and re-requested a line in the same cycle. At tick225842000 private L2
processed the new Store (`M -> M`) before the older PUTX (`M -> MM`), sending
stale data and forgetting the renewed L0 ownership. A later eviction then failed
in MM. See `diagnosis/writeback_v4`. The `rank=3` declaration did not set actual
input service order: generated `wakeup()` visits ports in source order.

V5 moves the return input before the regular input, so ready evictions are
processed before new local requests. The existing equal two-cycle L0 request
and response latencies plus ordered return FIFO preserve this dependency;
changing relative local-link latency/order would require separate validation.
No capacity, latency, or transition-throughput value changes in this repair.
`test_bounded_writeback.py` checks routing/transition structure and original
dynamic failure evidence; it does not replace full numerical simulation gates.

Failed builds/runs remain under `build/experiment`, `build_v2/experiment_v2`,
`build_v3/experiment_v3`, and `build_v4/experiment_v4`; traces/ablations are under
`diagnosis/`. Latest validation uses `build_v5/`, `experiment_v5/`, repeating
every gate. Status files are authoritative; v5 is not accepted until gates pass.

V5 validation update: all eight T512 gates passed, including exact legacy
extracted-counter equivalence, full arithmetic checks, 1024/4096 all-numeric
non-host-stat invariance, and four-message stress. Both T4096 reruns passed the
old initialization crash points and entered ROI; full target acceptance remains
pending. `experiment_v5/resource_audit/REPORT.md` reports accepted results only.

## Validation sequence

`build_hardware_bounds.py` preserves pre-edit sources and freezes the new build.
`run_hardware_bounds.py` runs at most two simulations concurrently, independent
of active queues:

1. T512 cache/SPM default-off controls: full independent arithmetic/export
   checks and exact extracted-counter/output equivalence to accepted controls.
2. T512 finite profile at global guards 1,024 and 4,096: full correctness,
   occupancy assertions, zero global-cap rejects and identical numeric non-host
   statistics across both guard settings.
3. T512 small-resource stress: LQ/SQ=8, TBEs=8/controller, queues=4 and targets=4.
   This is a stress configuration, not a proposed production design.
4. T4096 cache/SPM finite profile at both guards with the same checks.

Full configuration audits allow only listed bounds and run/guest metadata.
The independent mathematical reference, guest hash and expected SPMCP count
must pass; failures stop this queue. Prior runs remain untouched.

These checks validate the exercised SE FA path, not exhaustive coherence,
full-system HTM/cache-maintenance/context-switch behavior or a final paper model.
Global-guard invariance is necessary but not sufficient for physical accuracy.
After functional acceptance, controller-depth sensitivity and protocol stress
are required before using this as the paper's primary hardware configuration.
