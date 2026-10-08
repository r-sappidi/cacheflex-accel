# Bounded ingress bypass sensitivity experiment

Goal: measure **FA end-to-end latency**, not infer its improvement from summed
SPMCP waiting time. The queue is the actual L1's `l0_cntrl.mandatoryQueue`.

## Mechanism

`MessageBuffer.cacheflex_ingress_lookahead` defaults to zero. Only the L0 mandatory
queue is enabled in this experiment. Following a resource/protocol stall, the
existing SLICC stall handler can select a safe younger request from the oldest
`N+1` queue entries: the head plus up to N younger entries. Test N=0,4,16.
Selection does not remove blocked requests, change timestamps, create capacity,
or bypass any cache-resource checks. Every alternative is attempted through the
normal protocol. SLICC charges a successful stall-handler selection against the
existing 32-transition-per-cycle budget; there is no additional issue bandwidth.
The oldest request is retried first each new cycle. Alternative probes are bounded
per cycle, and normal input-port priorities remain intact.

The selected message stays in the original heap. Peek/dequeue/stall/recycle act
on that message consistently; removing it rebuilds the heap. Functional accesses
and capacity accounting still see every pending message. Host-side references
are scheduler metadata, not another modeled message/data buffer.

## Conservative ordering

Only ordinary cacheable LD/IFETCH and plain one-line SPMCP can bypass. Every older
request in the age window must be independent: no shared coherent source line,
and no overlapping SPM destination slot for copies. Stores and unsupported types
are full boundaries, as are atomics, LL/SC, uncacheable/strictly ordered accesses,
transactions, maintenance, multicast and DMA/batch copies. Stalled-map requests
are also checked conservatively. Existing CPU LSQ, bridge shadows, fences and
coherence mechanisms remain enabled. This is not a claim of exhaustive memory-
model verification; native predicate tests and simulator regressions are gates.

Search/selection has no separately modeled circuit delay or area. Failed probes
consume the existing controller budget. This is a bounded architectural
sensitivity experiment, not an RTL timing/area implementation.

## Fixed workload and hardware

Single head, d64, 8 cores, VL16; 64KiB L1I/L1D and private L2 per core, 4MiB LLC.
Existing translated MICRO credit pools and retention are unchanged. Fixed kernels:
cache Br32/Bc128/G2; SPM Br16/Bc128/G4/two ways. No kernel tuning, extra SPM backing,
DMA, multicast, enlarged request budgets, or network changes.

`scripts/run_ingress_bypass.py` uses a frozen build/config snapshot, then:

1. Three existing roundtrip/RAW/release-GETX protocol tests at lookahead16.
2. T512 cache/SPM at N=0,4,16, with independent full numerical checks and exact
   output equality to the fixed-kernel controls. N=0 must match all previous
   finite non-host statistics, excluding only newly introduced zero counters.
3. T4096 cache/SPM at N=0,4,16, with the same checks. Maximum two new simulators.

All configurations are audited to differ only in the declared queue parameter.
Failure stops this queue, retaining diagnostics; unrelated jobs are untouched.

## Read the result correctly

Primary: each backend's ROI speedup versus its own N=0, plus cache/SPM ratios
at each matched N. Also inspect successful bypass counts and SPMCP aggregates.

**Use ready-to-successful-forward ingress time**, the sum of FIFO waiting and
post-first-attempt forwarding waits. Lookahead may reduce time to the first
attempt simply by checking requests sooner; that alone is not a performance win.
The existing copy observer retains only live requests and running aggregates.
Its per-core lifetime sums must match independent outstanding-request integrals.

If no requests can bypass because all need the same full resource, the experiment
has not removed the waiting bottleneck. If dwell time falls but ROI does not,
the queue's waiting is largely overlapped or another bottleneck offsets the gain.
Failure to improve with this conservative selector does not rule out every
possible ingress scheduler or downstream optimization.
