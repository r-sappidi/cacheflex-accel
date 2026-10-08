# Why faster SPMLD does not accelerate FA

Experiment: unchanged real-staging T4096 single-head/VL16,8core FA. Compare
normal calibrated SPMLD response5 versus response1; identical kernel, SRAM,
SPMCP, credits, network, prefetcher and CPU issue configuration. The previous
fully validated full-ROI times are5908.1544us versus5921.8840us. Oracle staging
is explicitly OFF in both cases.

`CacheFlexDeps` is a default-off, bounded observational debug flag:

- IQ arrival, static/micro PC/disassembly, renamed source/destination IDs;
- actual producer-to-consumer wakeup edges and all-register readiness;
- memory response, register writeback/wakeup, ready-list entry and actual issue;
- a snapshot of remaining ready entries after each issue selection, labelled
  by observed same-opclass FU allocation failure or exhausted global issue
  budget (other reasons remain unclassified);
- architectural commit and blocked ROB-head identities.

It changes no scheduling decisions, resources or simulated delays. Copies of
ready queues are used only for observation. Source-register provenance covers
already-ready values; actual wake edges override reconstructed provenance.
Pinned/boundary/unknown cases are not silently treated as known dependencies.

The experiment first repeats full T512 math/export and exact non-host-stats
checks against both previous runs. It then samples a predeclared early interval
near first worker computation and a middle interval. A low-volume SPM_CPU scan
finds the first SPMLD in cores0/7, followed by a detailed window with2500cycles
of preceding context and30000cycles after both cores' first sampled loads.
Each detailed window is repeated with tracing OFF: every finite non-host stat
must match at exactly the same stop tick before analysis is accepted.

Outputs: `consumer_rows.json` with explicit load/consumer/blocker SNs, PCs and
ticks; `consumer_summary.json`; `{early,middle}_REPORT.md`. First committed
direct consumers are primary, so repeated reuse of one V vector does not
inflate its apparent opportunity. Boundary/reissued/incomplete records are
reported. Source readiness, ready-to-issue delay, and FU/issue-width waits are
separated. These request/instruction intervals are NOT additive FA runtime
fractions. Dynamic sequence numbers are not assumed identical across variants.

All runs remain under2new simulators,1GiB compressed trace/case,1hour/case and
50GiB free-disk guards. Previous simulations/builds/results and SwiGLU inputs
remain intact. Frozen binary/source/config/guest manifests accompany the runs.

Run `python3 scripts/build_spm_consumer_trace.py`, then
`python3 scripts/run_spm_consumer_trace.py`. Current output is
`results/cacheflex_flash_attention/spm_consumer_trace_20260923/experiment_v2`.
The earlier retained experiment passed both full T512 gates but its scan ended
before SPM computation began; it is not a numerical or simulator failure.

The immediate first consumer is `splice`, so that observation alone does not
establish when useful arithmetic can execute. Run
`python3 scripts/analyze_spm_fma_consumers.py <experiment> early` (or `middle`)
after the matching observer-equivalence gate. It follows actual dependencies
through lane assembly to the first FMA, and follows later-ready splice inputs
back to coherent loads or an earlier FMA. Earlier FMA roots are accumulator
dependencies, not a claim about the ultimate source of their delay.

Early traces show a median 7-cycle interval between successive-slot SPMLDs
even when access latency falls from9 to5cycles. The same cadence is present at
IQ dispatch. `run_spm_frontend_trace.py` waits for the initial queue, then adds
existing Fetch/Decode/Rename observations to the exact early prefix to inspect
upstream instruction delivery. It changes no hardware; all non-host stats must
again match the existing trace-off prefix. This is a follow-up diagnostic, not
yet evidence for a particular front-end optimization or its runtime benefit.

Completed frontend follow-up: `analyze_spm_frontend_trace.py <experiment>`
produces `frontend_followup/REPORT.md` and raw-tick/SN event evidence. Both
variants pass exact timing invariance. The steady 19-instruction/76-byte loop
is fetched in a5/5/5/1/0/3/0 repeating pattern; the two empty cycles are at
fetch-buffer refills. Median fetch, dispatch and SPMLD-issue intervals are all
7cycles in both variants. This is a local instruction-supply throughput limit,
not proof of a whole-FA bottleneck fraction. RubyPort same-tick responses pass
through PacketQueue's minimum curTick()+1 scheduling; audit that clock-edge
interaction before interpreting the two-cycle refill behavior as real silicon.
