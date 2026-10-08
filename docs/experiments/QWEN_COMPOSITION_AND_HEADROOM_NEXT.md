# Qwen composition and targeted headroom work

Related AI-10/11 measure FFN/projections; L3-06..10 measure attention;
AI-03/04 demonstrated first-use/prepacking confounds; MOD-09 audited hardware
bounds. This work prepares accepted-only accounting and diagnostic designs.
No new simulator workload is launched. It is not new performance evidence.

## Component boundaries

One decoder layer: RMSNorm -> QKV (+bias, head layout) -> RoPE -> FA -> output
projection (+head merge) -> residual -> RMSNorm -> SwiGLU -> residual.
QKV widths896/128/128, output896->896, FFN896->4864->896. Layer count24.
Count QKV bias/layout and output head merge inside the projection measurements;
FA tile packing/staging/normalization/drain inside FA; SiLU and elementwise gate
product inside SwiGLU. Two RMSNorms and two residuals must be measured separately.
RoPE follows the QKV head-layout conversion and must preserve that boundary.

Model extras: embedding, final RMSNorm, vocabulary projection. Working scope is
last-position logits for prefill followed by generation, vocabulary151936;
all-position logits is a different workload and must have a separate result.
This is a proposed accounting convention, not an already measured model policy.

Use first-invocation times consistently; do not insert repeat-call FFN or
projection times as if they were matched prepacked model execution. A prepacked
policy needs a separate matched measurement. Synthetic FA inputs currently do
not come from the projection guest; integrated-layer correctness and cache-state
validation remain required. Mixed operator arithmetic must be stated: FA FP16
GEMMs/FP32 online state, FFN/projections FP32 accumulation with FP16 storage.
No BF16 checkpoint-quality or integrated-execution claim.

scripts/collect_qwen_components.py consumes only accepted.json-backed records,
keeps missing fields null, records source hashes and selection status, and does
not emit model/layer totals. It can refresh while searches run. SPM+L3 inherits
private-SPM non-attention times; no fabricated FFN/projection L3 benefit.
T16K/T32K model totals cannot be inferred from short FFN/projection measurements.

When all pieces exist, validate sum(component latency) against at least one
integrated layer with identical precision/layout/packing and explain discrepancies.
Only then form24layer projection plus measured extras, with explicit activation/
cache-state assumptions. Baseline and optimized totals are summed separately;
do not average component speedups. Energy is a separate unmeasured quantity.

## Narrow engine sensitivity (not queued)

MOD-09 found low compulsory ALU-issue duty. L3-02/03 used16lanes and already
implemented bank pipelining; this proposes right-sizing on the current Qwen
winner, not rediscovering that pipeline. Freeze T4096 guest/Br16/Bc256/R256,
serializedFP16/init engine, NoC and direct comparisons; retain existing controls.
Hypothetical lane schedules, assuming unchanged3cycle multiply/add/conversion:
16lanes: arithmetic9, conversion6, initiation4cycles;
8lanes: arithmetic13, conversion10, initiation8;
4lanes: arithmetic21, conversion18, initiation16.
These are scheduling hypotheses, not characterized clock/area claims. Timings
must be represented consistently in all related scheduling reservations. Keep
queue capacities, SRAM ports, coherent traffic, CPU issue and clocks unchanged.
Decision: does a smaller engine preserve lower complete FA ROI than privateSPM,
and how much incremental latency is lost? No area ratio inferred from lane count.
Run arithmetic/mixed/ordering gates before FA if modeling code changes.

## Source admission and completion criticality (not queued)

L3-02 retry isolation and L3-09 direct transport already address distinct pieces
of this path. Busy/full retry counters cannot establish removable latency.
First produce a time-resolved sample identifying source-buffer-full intervals,
oldest live command, core progress, and completion latency on the current winner.
Use a matched timing-only instrumentation equivalence check before interpretation.
Then change one documented timing component while retaining finite storage,
source data capture, per-row order, completion and barrier visibility. Do not
pretend sourceACK=1cycle removes source-buffer lifetime or globally add credits.
Decision: lower full FA ROI, not fewer retries or shorter command lifetime alone.
An optimistic result is a headroom diagnostic; an implementable variant still
needs a feasible data path and hardware cost. Exact variants remain to be specified
from trace evidence; no fake ready-to-run oracle has been prepared here.

## Bank-service sensitivity (not queued)

L3-03 tested pipeline overlap; MOD-09 distinguishes50cycle read/convert/arithmetic
from35cycle post-install response. A future test must separate bank read latency,
ALU scheduling, write-port occupation and return delay. Current l3_spm_latency
couples access and response delays, so merely changing it is a bundled sensitivity,
not isolated bank-read attribution. Shared SRAM changes require matched cache/
private arms; reduction-only latency changes must stay within credible physical
assumptions. Decision: benefit survives conservative costs, or meaningful optimistic
headroom warrants a design change. Leave frozen production queues unchanged.
