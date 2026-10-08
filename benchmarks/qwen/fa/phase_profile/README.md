# FA-49: passive per-core phase and memory-overlap profiling

User requested architectural guidance from matched cache and ordinary SPM phase
ROIs. Reuse exact FA48 guest/labels, R256/Br16/Bc256, T256 gate then T1024 Qwen
14Q/2KV,d64,8cores. NCC mode0: no PV engine, idealized service or operand probe.
Four arms cache_off,spm_off,cache,spm; ON/OFF differ only in host observer env.
Every non-host statistic/output bit must match OFF; SPM must additionally match
accepted FA48 spm_inert at both sizes. No retuning or new guest compilation.
This is a matched diagnostic guest, not a newly established best baseline.

One workbegin/workend interval remains the outer ROI. Existing assembly labels
trigger passive commit-PC events. Core phase times partition that interval.
Phases include K/Q packing, staging/loop setup, QK, softmax, P packing, fused
CPU PV+recurrence, normalization, release and barriers. Fused PV+recurrence is
kept fused. No global reset/dump on phase transitions; simultaneous cores retain
independent labels. The observer is isolated in pseudo_inst, Sequencer and LSQ
objects; no modeled queues, timing, coherence or guest instructions are changed.

For each phase/core record:
* Sequencer SPMCP issue/completion and in-flight union/occupancy integral.
* SPM-load RAW block start, successful retry or squash; waiting-load union and
  occupancy integral, completed wait ticks checked against existing RAW stats.
* Accepted coherent/SPM data-load packets and response lifetimes, bytes, union
  and occupancy; physical traffic includes speculative packets. Requests sent
  before ROI are excluded; unfinished packets at ROI end are explicitly counted.
* Load port-rejection counts; SPM counts reconcile to existing hardware counters.

Copies are tracked after Sequencer issue, not their earlier CPU decode/LSQ wait.
Software staging phase measures loop execution; it need not cover completion.
Request event phases use the core's current retirement phase, which may differ
from the originating instruction under out-of-order execution. A pending request
or blocked load does NOT imply the whole core is stalled. Intervals overlap;
never add their durations to phase time or sum across cores as wall latency.
Copy occupancy integrates number outstanding; busy time is their union. Service
latency sums overlap and are not removable runtime. RAW squashes are reported
separately from completed waits. Per-core timelines/last arrival and full ROI are
reported alongside averages. Remaining outer-ROI boundary overhead is explicit.

Acceptance: full independent oracle, historical/peer output bits, exact tasks,
copies, config, LSQ bounds; phase work counts and exact timeline partition;
issued/completed/pending conservation, copy-latency/integral equality, RAW counter
reconciliation and exact OFF/ON non-host stats. Six host tests cover overlapping,
phase-crossing, squashed and censored lifetimes plus same-guest plans/work.
Initial8 cases. Max4simulators,10GiB initiallease/2CPUs each; shared20%CPU/RAM,
64GiB host headroom and progress watchdog. Llama unchanged. No T4096 auto-promotion.

Validated outcome: all8 cases complete. T1024 cache1000.4952us/SPM950.7684us;
OFF/ON exact all non-host stats and outputs,2,292frozen hashes checked. See
results/cacheflex_flash_attention/phase_profile_20261005/REPORT.txt plus CSV,
JSON and stacked per-core chart (PNG/SVG/PDF). Softmax33.24%,PV+recurrence22.42%,
QK20.85%,staging3.04% of SPM outerROI on the final-arriving core. Zero blocked
SPM RAW loads; copy-in-flight intervals overlap staging and a small part of QK.
Coarse T1024 task imbalance is visible in barrier time; no4K or globally optimal
baseline claim. Phase grouping in the chart is accumulated, not chronological.
