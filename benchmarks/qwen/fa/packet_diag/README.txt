FA-42: identify why DATA latency rises under private V staging

Historical evidence: FA-13/16/18 traces established selected copy/admission
interference on older profiles. FA-15/20 widened all traffic, FA-21/40 idealized
many copy costs together. None separated source-NI buffering, individual router
waiting and receiver ejection for current Qwen copy versus coherent DATA.
The R062/R006 route audit rules out route-mixture-only effects in that historical
pair; this experiment uses retained-protocol FA-39 controls, not those old timings.

No new modeled resources or latency shortcuts. Recompile ONLY NetworkInterface.cc
against frozen retained sources/headers, preserving every other linked object.
Add debug-only copy/coherent response metadata, read-only NI readiness/credit/
ordering snapshots and receiver stall labels. Existing per-flit router events
identify readiness, grants, no VC, no credit and ordering. No protocol/guest change.

Ten serial cases: full accepted T256 SPM replay, trace off/on (whole small ROI),
then accepted T4096 cache/SPM prefix pairs trace off/on at 100us after ROI entry
and at 40% of each backend's ROI. Each 4K window is 10us with 4ns drain past trace end.
Same normalized phase is not identical dynamic work. Complete packets only;
report censored packet counts. Prefixes are diagnostics, never timing results.
No full matrix or duplicate full-ROI tuning. Existing accepted full controls reused.

Acceptance: exact T256 output and all finite nonhost stats versus original, exact
config (output paths only), exact trace-on/off nonhost stats at each bounded stop,
input hashes, explicit packet interval conservation, nonempty classified DATA;
SPM windows require actual copy DATA. A missing class is a failed diagnostic,
not permission to infer no contention. Original evidence and failed runs retained.

Packet tail-flit decomposition: source ready-to-flitization, NI creation-to-tail
send, router pipeline, router ready-to-grant, crossbar/link transport and receiver
arrival-to-enqueue. Source ready wait is part of source message age, not additive.
NI tail wait includes serialization and upstream effects of earlier packet flits.
NI and router block event samples are not additive stall cycles or runtime fractions.
Copy DATA classified by ResponseMsg.DstSPMAddr; ordinary DATA kept separate.
Other message types still recorded. No address-only transaction matching.

Decision: use repeatable localized intervals/block events to choose one justified
NI, router/VC or receiver change. Existing trace-only evidence cannot establish
its runtime payoff. Require later matched full-ROI improvement with unchanged
arithmetic/config except that change, and cache sensitivity control, before
claiming an SPM-specific speedup or proposing hardware resources.

Resource policy: shared 20% CPU/RAM broker, two allowed host CPUs, one simulator;
6GiB compile reservation, 24GiB link reservation, 10GiB per run (8GiB initial
simulator AS plus 2GiB coordinator), admitted 4GiB growth, >=64GiB host headroom.
2GiB compressed trace cap, >=50GiB disk reserve, 24h per-command limit. Pressure
pauses and resumes only owned child. Existing queues and frozen files untouched.

Launch: python3 benchmarks/qwen/fa/packet_diag/launch.py
Status: results/cacheflex_flash_attention/packet_diag_20261004/experiment_v1/status.json
Build: results/cacheflex_flash_attention/packet_diag_20261004/build_v1/status.json
Analysis: each trace run packet_analysis.json; REPORT.txt only accepted samples.

Parser coverage note: only RequestMsg/ResponseMsg packets have NI_CLASS metadata.
The v1 censored_packets count includes complete packets of other unclassified
message structs as well as boundary-incomplete packets; it is an exclusion count,
not a pure trace-boundary count. All DATA comparisons use classified responses.
