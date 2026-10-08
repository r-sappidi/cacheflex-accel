FA-43: retained-protocol Qwen network-contention causal sensitivity.
Prior evidence: FA-42 localizes copy DATA tail waiting to router VC/credit flow
control; FA-15/20 widened all links on older profiles; FA-21/40 alter broad copy
service. Inventory search found no buffers_per_data_vc experiment. New experiment,
not retuning. Same FA39 guest/simulator/math/Br16 Bc256 R256, Hq14 Hkv2 d64.
Reuse accepted FA39 T256 and T1024 controls (including valid controls in a queue
that later failed an unrelated packed-kernel case). Require frozen output hashes.
Four parallel arms: cache/SPM x data VC depth8/16 versus existing4. Each T256 full
oracle gate precedes T1024 full ROI. No link/clock/routing/VC-count/credit-pool or
kernel change. Data-vnet buffering changes all messages on those vnets; it is not
an SPM-only change. Added buffer capacity is explicit, not free hardware.
Source run.py derived from FA39 run.py with original scientific validation intact,
plus exact historical output bits. Strict config permits ONLY named data depth
and original backend/guest settings. Sources/control records hashed before runs.
Shared broker10GiB/arm (8GiB sim AS+2GiB coordinator), admitted growth; four sims,
eight CPUs within original24-of128 CPU budget,20% RAM and64GiB host headroom.
Separate status roots and reporter outputs prevent races. Gates before targets.
Observe full ROI, network/queue latency, flits and copy work. Meaningful evidence
requires reduced targeted latency AND reduced full ROI; compare own baseline
fractional gains for cache/SPM. No stall sums or exact runtime attribution.
Decision: >=2% SPM latency gain plus positive excess over cache motivates T4096
confirmation; smaller/null results retained. T1024 outcome does not prove4K.
Response-VC changes and copy-specific contention oracle remain contingent
follow-ups; NOT implemented or queued here. Do not call depth relief an oracle.
