FA-46: overnight near-cache operation screen

Prior evidence: FA-39/SYN-01 implicated coherent K/packing, but samples are not
critical-path fractions; FA-40/43/44 found limited gains from SPM service/network
changes. MOD-11 modeled standalone PV schedules, not full-kernel execution.
This experiment supplies full-kernel operation-substitution evidence. It is NOT
an implementation of an asynchronous near-cache engine.

Qwen T256 correctness gates -> T1024 screen, all14Q/2KV heads, 8cores, d64,
Br16/Bc256/R256, finite retained hardware and ordinary V-only placement.
Same new ELF controls. No changes to old guests/models/queues. Four workers,
2allowed CPUs and10GiB lease per worker,8GiB initial simulator AS + admitted
growth. Aggregate20%CPU/RAM and64GiB server headroom via existing broker/monitor.

Arms:
cache,spm: original kernel, command path inactive.
drained: ordinary SPM kernel plus per-PV DSB; diagnostic synchronization control.
kpack: K128x64 transpose via functional proxy; timed coherent source read and
       destination read/write probes, DSB ordering, command overhead retained.
ppack: P16x256 eight-row packing via same memory-probed proxy.
pv_oracle: zero-service functional PV, source transfers untimed; deliberately
           optimistic opportunity test, NOT pure arithmetic-only attribution.
pv_traffic: same paired-kernel P and SPM V operand-load instructions (FMLAs
            removed), drain, functional FP16 PV, CPU FP32 recurrence retained.
pv64/128/256: pv_traffic + blocking serialized service costs of5472/3424/2400
             core cycles per16x256x64 operation. Assumes1FMA/lane/cycle;
             32command cycles, P/V/output32B/cycle budgets added in series.
             Deliberately conservative overlap but optimistic operand availability.
pv_l3_budget: pv128 +2728cycles (43008B/16B-cycle +40cycles),6152cycles total.
              This is delivery-cost sensitivity, NOT actual L3 placement/NoC.

Functional proxy reads real claimed private SPM V, uses SoftFloat nearest-even
FP16 FMA in sequential key order, restores rounding state; actual input/output
arrays, no offline answer substitution. CPU retains scaling, FP32 recurrence,
softmax, QK, normalization, staging, core ownership, finalization. PV output is
functionally written to coherent scratch, CPU consumes via normal loads.
Functional writes do not model engine stores/coherence ownership; finite output
service budget substitutes for these costs, with CPU-visible cache effects still
from functional access. P/V traffic probes preserve instruction request sequence,
but DSB and arithmetic removal change overlap and occupancy; not a strict bound.
No bank arbitration, finite engine queue, or real engine SRAM implementation.
One outstanding blocking operation/core, bounded16x256P/256x64V/16x64output
functional temporary buffers. These host vectors are not a claimed SRAM design.
No cross-core engine contention in proxy. L3 cannot be declared feasible from it.

Acceptance: full independent online reference, dense T256, exact historical bits,
exact per-core work, copy requests=completions, no reductions, strict full config,
finite queues, hook call/service counts, cross-arm exact outputs; input hashes.
Packing permutations/work/service counts have four host tests. Numerical gates
must pass before target use. Failures do not produce timing claims.

Automatic policy: screen11arms, max4simulators. Promote at most2 non-oracle
candidates with>=5% full-ROI gain over matched SPM toT4096, with cache/SPM controls.
5% is a triage threshold, not a publication standard. A null result with this
blocking proxy does NOT rule out an asynchronous design. A positive oracle with
negative traffic/finite results points to delivery/service costs, not proven value.
REPORT.txt and analysis.json update after arm completion; final scientific review
required. Controls and finite cases may finish overnight; ETA is not guaranteed.
