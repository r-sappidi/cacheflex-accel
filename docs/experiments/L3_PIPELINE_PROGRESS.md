# Bounded L3 reduction pipeline and FP16 contribution transport

## 2026-09-28 — implementation

User requested implementing bank pipelining first, then compact contributions,
and measuring incremental/combined end-to-end latency. Reference ledger L3-02;
do not revisit extra SPM backing or treat queue/stall cycles as removable runtime.

Starting point: accepted reduction retry isolation. T1024 async147.3932us,
T4096 async1173.7940us (original serialized engine, FP32 payload). T4096 is
0.475% slower than matched private1168.2400us; still no net gain demonstrated.
Original frozen builds/guests/results and unrelated active queues preserved.

1. Opt-in bank pipelining: at most the existing four captured commands in
   flight per bank; same TBE/data storage, same shared64B/cycle SRAM port,
   read35cy and arithmetic9cy. One16-lane pipelined multiply/add datapath,
   minimum4cy initiation interval. Reads/writes arbitrate the original shared
   service port. FP32 row locks persist until install. A future ALU reservation
   also serializes its lane beats correctly for mixed FP16/FP32 arrivals.
   This changes the engine organization, not merely its bookkeeping limit;
   timing is an explicit design assumption, not an RTL-calibrated specification.
2. Typed FP16 contributions: bit58 selects64 densely packed half values,
   widened at the bank using SoftFloat. Existing FP32 accumulator and separate
   multiply/add rounding unchanged. Adds a modeled16-lane conversion pipeline,
   3cy latency +3 lane beats =6cy before arithmetic. Real network message144B
   including16B header, versus272B previously. No changes to network bandwidth.
   For compatibility the CPU-to-source interface still captures a full256B
   vector; upper128B are ignored by this explicit typed command, not compressed
   arbitrarily. Destination hazards cover256B despite128B network payload.

New simulator root results/cacheflex_l3_spm/pipeline_20260928. build_v1 was
interrupted before validation to add mixed-type ALU reservation; its source and
logs retained. build_v2 is the intended candidate. New guest_v1 has same-ELF
cache/private/FP32-reduction/FP16-reduction modes. No changes to QK/PV math,
softmax, tile shapes, allocation, initial zero upload or per-core mapping.

Validation plan: old-ELF default exact controls, pipelined directed tests,
old-ELF pipeline-only FA, typed/mixed directed tests, then same-new-ELF
serialized/FP32, pipelined/FP32, serialized/FP16, pipelined/FP16 plus cache and
private-SPM controls at T1024 and T4096. Independent full numerical oracle,
exact output hashes, command/payload/occupancy checks, config-delta audit.
Maximum two new simulators; stop this queue on failure and retain evidence.

Build_v2 completed SHA625a38c8b126bc0a805061e28d90595510d8120a6d1d9eaf5a021c6d26ccd80c.
26 static/source tests pass. guest_v2 includes additional mixed FP16/FP32
directed test; guest_v1 retained. experiment_v1 failed before launch due to a
reference helper manifest using scripts/ instead of the module's actual path;
fixed without changing numerical checks. experiment_v2 PID1286011 launched
the old-ELF exact cache/private controls; runtime acceptance pending.

First acceptance: all three old-ELF controls reproduce all finite nonhost stats
and exact outputs. Pipeline directed async/sync/one-entry tests pass1040commands
each, zeroerrors. Two overlapping reductions observed in nominal smoke; one-entry
stress max1. Old-ELF T1024 pipeline accepted147.1656us versus147.3932us serialized
(0.1544% lower latency). Full oracle/hash/config/copy/payload checks pass;8192
commands, maximum3 overlapping bank reductions. Overlap is real but has minimal
end-to-end effect in this case. T4096 pipeline and typed/factorial stages pending.

Old-ELF T4096 pipeline accepted1175.2736us versus1173.7940us serialized:
0.1261% higher latency, maximum2 concurrent bank operations,32768 commands.
Thus pipelining alone does not improve FA materially at either tested size.
Typed directed serial/pipeline/mixed/one-entry tests pass exact arithmetic,
1040commands each (mixed520FP16), finite bounds and wire totals:149760B allhalf,
216320B mixed, versus282880B allFP32. These directed timings include different
guest code and are NOT FA improvement claims. Full matched FA stages follow.

## T1024 factorial accepted

All26 source guards,3 exact controls,8 directed tests,2 old-ELF pipeline FA cases,
and all6 matched-new-ELF T1024 cases pass. Every FA output independently checked
and bitwise-identical to its accepted source. FP16 representation changes neither
the FP32 accumulator nor the mathematical result.8192 reduction commands each.

| Variant | ROI us | Reduction vs serialized FP32 |
|---|---:|---:|
| Cache |119.6668|not applicable|
| Private SPM |115.5796|not applicable|
| Serialized FP32 reduction |147.2108|0%|
| Pipelined FP32 reduction |147.0968|0.077%|
| Serialized FP16 contribution |143.8136|2.308%|
| Pipelined FP16 contribution |141.1488|4.118%|

Incremental FP16 after pipelining saves4.044%; pipeline after FP16 saves1.853%.
The two interact: bank pipelining matters more after compact transport than on
the original FP32 path. This is a measured conditional effect, not proof that
all remaining latency is network-related. Combined remains17.952% slower than
cache and22.123% slower than private SPM; no overall FA win atT1024.
Baseline->combined total network flits561997->496254, command mean lifetime
425.536->308.831cycles, source-full attempts463145->325114. Do not interpret
these differences as additive portions of end-to-end latency. Compact mechanism
also removes CPU widening (instructions4089403->4048443), so not network-only.

T4096 same-new-ELF cache/private pair active (2610469/2610424 at launch), followed
automatically by serialized/pipeline/half/both. Queue1286011, max2 simulations.
Accepted-only incremental analyzer watches and writes EFFECTS.md/effects.json.
