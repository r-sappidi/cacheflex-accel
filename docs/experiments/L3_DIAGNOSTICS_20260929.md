# MOD-10: Qwen L3 latency and engine-size diagnostics

User authorized setup and launch on 2026-09-29. Prior evidence: L3-02 retry
isolation improved older kernels but did not isolate all command-critical delay;
L3-03 pipeline was near-null on older single-head kernels; L3-04 initialization
helped but finalization regressed; L3-09 direct transport helps current Qwen by
only0.175% at4096. MOD-09 audited finite timing/resources. These are existing
mechanisms. New work separates reduction read latency from completion delay,
observes the current source path, and transfers engine timing/pipelining to the
frozen current Qwen winner. No FFN L3-offload implementation is included.

## Frozen workload and comparisons

Use existing exported guest and all inputs unchanged: Qwen causal14Q/2KV heads,
d64, T4096, Br16/Bc256/R256 for reduction. The primary source is L3-09's accepted
direct9037.8088us; matching NoC9053.6880us, private9473.3184us and independently
selected cache10527.9640us remain preserved. Full every-output independent
reference and exact prior-output bits are required. Only modeled timing changes;
no relaxed tolerances, reordered arithmetic, new queues/credits, wider ports,
extra data SRAM, changed CPU issue widths or guest changes.

## Interventions

New controller fields l3_reduction_read_delay / l3_reduction_reply_delay default
to0, meaning inherit l3_spm_latency (35Ruby cycles). Read override applies only
to ordinary reduction commands; initialize still bypasses the read. Reply
override applies only after normal reduction installation, where the old35cycle
completion delay occurred. It does not acknowledge unfinished results. Ordinary
shared SPM/coherent accesses and finalization keep their original timings;
both read/write64B-per-cycle port charges remain. All queues and hazards remain.

| Profile | Reduction read | Completion delay | Arithmetic / conversion / II |
|---|---:|---:|---|
| default |35|35|9 /6 /4|
| read_fast |1|35|9 /6 /4|
| reply_fast |35|1|9 /6 /4|
| both_fast |1|1|9 /6 /4|
| lanes8 |35|35|13 /10 /8|
| lanes4 |35|35|21 /18 /16|
| pipeline |35|35|9 /6 /4, existing pipeline enabled|

Fast cases are deliberately optimistic diagnostic interventions, NOT feasible
SRAM/response timing claims. They preserve work/bandwidth/ordering but grant a
reduction-only latency advantage to test headroom. Any realizable shared-SRAM
change must later also be applied to cache/private controls. Narrow-engine
profiles model scheduling at fixed pipeline depths; no physical lane removal,
area, energy, or frequency result is claimed. Pipeline is an explicit
revalidation of L3-03 on a different guest/lifecycle configuration.

Passive CacheFlexReductionTrace logs capture/issue/completion events and source/
flight occupancy plus oldest age every64 core cycles while the service event is
active. No extra simulated event or timer state. Restrict trace to100us near
the start of ROI and128MiB maximum. Default small-Qwen trace-on must exactly
match old trace-off full numeric nonhost stats and output; repeat this on4096.
Sampling is conditional on service activity, not a stall-time denominator.
Core commit progress remains in the ordinary run log. Raw traces preserve core,
tick and command serial. Event ages do not establish a removable runtime share.
CPU capture ACK is already one cycle; this experiment does not invent a faster
source-capture datapath. Reply sensitivity tests completion-credit lifetime;
source issue interventions require evidence from these traces first.

## Gates and queue

19 gates precede8 targets:
3 exact old-default cache/private/NoC small-Qwen controls; direct exact and
trace equivalence;2 optimistic no-op cache/private exact controls;
6 fast directed FP32/mixed/init/both/sync/one-entry-slow cases;
3 narrow4 mixed/sync/one-entry cases (existing directed sources also cover pipeline);
3 small-Qwen fast/narrow4/pipeline full-output gates.
Then T4096 exact direct+trace,6 direct timing profiles, and NoC both_fast.
No extra run is needed for pre-existing accepted full cache/private controls;
new-build small defaults plus full direct exact replay establish local regression
evidence. Existing directed guests and numerical tests are reused, not rewritten.

Decision: compare whole FA ROI against the identical-kernel source and private
SPM, retaining null/regressive outcomes. Read/reply factorial measures separate
effects and interaction; neither retry counts nor lifetimes substitute for ROI.
If even optimistic delays buy little, deprioritize that path. Narrower timing
profiles ask how much performance is lost while retaining private-SPM benefit;
they do not predict silicon savings. NoC both_fast checks whether combined
headroom depends on the direct link. Any speedup interpretation remains scoped
to these models, and any feasible follow-up needs hardware justification.

## Resources and preservation

Isolated results/cacheflex_l3_spm/diagnostics_20260929; previous source copies in
preserved/. New build_v2 is separately linked/frozen; old simulator/config/
guest/reference artifacts remain untouched. Build uses4 low-priority workers
and waits for >=50%CPU idle, >=128GiB available RAM, zero swap use, >=100GiB disk.
Queue runs at most1 additional simulator and6 total, with the same launch guard.
It pauses ONLY its own simulator if a launch race exceeds6 or idle<25% persists
for3 resource samples, resumes under the launch thresholds. Runtime RAM64GiB,
disk75GiB and96h/case guards fail closed. Any validation failure stops subsequent
diagnostics; evidence is retained. Existing production queues stay unchanged.
