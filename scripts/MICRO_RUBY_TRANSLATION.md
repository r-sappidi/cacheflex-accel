# MICRO-derived Ruby resource translation

Status: implementation under validation; not a claim of cycle-identical Classic
reproduction or calibrated physical hardware. Prior finite-resource model and
its accepted results remain under hardware_bounds_20260921.

## Source of truth

Pristine MICRO checkout: results/cacheflex_flash_attention/micro_upstream_20260918,
commit 82ac0b9de1ed408ce5cd53067267c8a855b4eef3. Read Caches.py, cache/base.cc,
cache/base.hh, cpu/o3/lsq{,_unit}.{cc,hh}, and
experiments/end2end/scripts/sweep_vl16.sh. The dirty sibling checkout is not the
reference. No cancelled MICRO reproduction benchmark is restarted.

Current simulator: build_v5, SHA256
defcf8502ebcab693d36ba06b3628a702b54c921e287d572b767e911f0e0cdb5.
Current validation queue: experiment_v6. Earlier numbered experiments preserve
failed/partial translation diagnostics and must not be substituted for this one.

experiment_v6 is complete: all12 cases passed. At T4096, cache4703.5076us
and SPM4751.4864us; both passed independent full-output checks. At T512 and
T4096,1024/4096 sequencer guards produced identical finite numeric non-host
stats, with no global-admission-full rejections.

Three instruction regressions now pass:64-round copy/load/store/writeback with
slot reuse; coherent RAW after writeback without a barrier; cross-core release
versus GETX snapshot race. The new tests exposed and repaired old SPMWB metadata,
completion and destination-ordering bugs. Physical coherent destination tracking
now lives in the existing SPM shadow entry; regular SQ entries still release
early. Barriers observe committed pending writebacks and pure SPM stores. SPMCP
slot operands are never forwarded as coherent store data. Default FA timings
and extracted counters still exactly reproduce the old accepted controls.

## Mapping

| Resource | MICRO | Ruby translation |
|---|---|---|
| Regular and SPM queues | LQ32/SQ48 in each domain | Existing dual LSQs, extra bounded-tracking policy OFF |
| Bridge tracking | Regular SQ releases on dispatch; SPM shadow persists | Existing bridge coordination retained; no new aggregate12-copy cap |
| CPU ports | Cache2load/1store; SPM1load/1store | Same issue limits in translated profile |
| L1I and L1D misses | 8 MSHRs each | Separate per-core miss credit pools at Ruby L0 |
| Bridge forwarding at L1 | Both hit and miss append to writebacks, return hit | Shares L1D8 write-buffer credits; releases on privateL2 consumption |
| Private L2 misses | 20 MSHRs | Per-core20-credit pool at Ruby L1 |
| Shared LLC misses | 32 MSHRs | One32-credit pool shared across address-interleaved LLC banks |
| Writebacks | L1D8, privateL2 8, LLC16 | Separate credit classes, not charged as misses |
| Pure SPM ingress | Separate SPM port, bypass coherent L1 | Separate queue/forwarding lane to privateL2; no coherent miss admission gate |
| Coherent port blocked | Full MSHR/WB resources block CPU-side admission | L1 admission gated by its own pools; forwarding checks privateL2 pools |
| Global sequencer cap | No Classic equivalent | Safety guard1024, verify1024/4096 identical timing/counters |
| Cache data arrays | Existing capacity repurposed | No new SPM backing or extra data SRAM |

Credits allocate only on relevant protocol transactions and release when their
buffered transaction completes. Coherence-only TBEs remain separate bookkeeping.
L0 eviction credit releases when privateL2 receives its PUTX. Local eviction and
recall responses retain their original FIFO ordering; pure SPM has a separate
input, not the previous finite-profile shared return lane.

## Deliberate differences and outstanding fidelity work

- Ruby still has directory transients, Garnet arbitration and distinct controller
  pipelines. Source-equivalent resource counts do not imply equal hit/miss latency.
- The MICRO script enables Stride at L1, AMPM at L2 and BOP at L3. This resource
  profile retains the existing Ruby prefetch configuration; it does not claim
  equivalent prefetch algorithms at all three levels.
- The separate SPM forwarding adapter still uses the L0 controller's transition
  budget, although it performs no coherent allocation. It is not a literal new
  independent physical controller or a claim of unlimited bandwidth.
- Ruby's privateL2/LLC writeback credit currently persists until protocol
  completion/acknowledgment. Classic can release its write buffer on downstream
  acceptance; this is a conservative lifetime difference to audit with results.
- Classic target counts are16/12/24. The implemented sequencer guard bounds
  same-line targets while an L1 miss is live. Ruby lower levels serialize or
  coalesce by requesting controller, not by Classic target entries. Lower-level
  target equivalence and prefetch-reserved MSHRs are not established.
- Full LLC miss capacity currently stalls new allocations, not all cache hits;
  Classic blocked-port behavior is not completely duplicated there.
- Protocol TBEs retain their pre-existing256-entry limit; local transport queues
  retain their existing configuration. They are not asserted to be physical
  infinite hardware or calibrated buffer capacities. Verify occupancy/guard
  sensitivity before interpreting performance as fully resource-matched.
- The repository's shadow table reserve(512) is allocation preallocation, not a
  hardware capacity. Do not label it a512-entry physical buffer. This profile
  follows the executable reference rather than silently introducing a new cap.
- The paper describes SPMWB placeholder-load tracking; the executable MICRO
  source uses bridge shadow tracking. The directed full-line SPMWB regressions
  pass after the fixes above; sub-line writes, arbitrary speculation/races and
  full-system context-switch correctness are not established by those tests.

## Evaluation

build_micro_resource_model.py creates a frozen simulator/config/source bundle.
run_micro_resources.py starts with default-off equivalence, full-reference T512
cache/SPM checks,1024/4096 safety-guard comparison, and two-credit stress. Only
then run matched T4096. Every accepted run requires independent arithmetic,
export/guest hash agreement, exact copy count and configuration audit. Failures
stop this queue, not unrelated frozen async FA or SwiGLU experiments.

First-stage measurements retain the existing single-head eight-core VL16
64KiB privateL2 FA geometry and LLC-retention setting. This isolates resource
translation; it is NOT the paper's512KiB privateL2 setup. Keep later geometry
comparisons separate. Report end-to-end ROI latency; resource full-check counts
are observations, not additive causal cycles or an achieved speedup.

`run_micro_resource_geometry.py` waits for all initial gates/measurements, then
uses the same guests with512KiB private L2 and no LLC retention. It is a separate
comparison, not a reuse of the small-cache controls. The fixed kernels are not
retuned to the changed model. Paper-quality best-to-best tuning remains later.

Current geometry output is experiment_paper_geometry_v2. The original output
preserves a pre-simulation MemorySize override type error; the launcher now
quotes string-valued Python overrides and reruns all gates without changing
the simulator or kernels.
