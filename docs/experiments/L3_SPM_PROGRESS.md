# Shared L3 SPM implementation log

## 2026-09-28 — scope and starting point

Requested: implement shared physical-L3 scratchpad, without reduction, and
validate functionality, arithmetic correctness and performance sanity.
Existing runs and frozen inputs must remain untouched. No additional data SRAM.

Related ledger evidence: FA-08 tested coherent LLC retention, not shared SPM;
FA-25–27/FA-32 tested private-L2 output residency/reduction. This experiment
instead repurposes LLC slots and makes them accessible across cores. No
performance benefit is assumed. Ruby L2Cache denotes physical shared L3.

Implementation sequence:
1. Preserve source/configuration provenance and establish default-off controls.
2. Add disjoint shared-slot namespace and finite network-routed accesses.
3. Reuse LLC eviction/recall before claim; preserve coherent source freshness.
4. Validate producer/consumer, dirty owners, slot reuse, writeback, vector masks,
   finite-buffer progress and cross-core synchronization.
5. Run latency/bandwidth sanity checks and a small checked FA comparison.

Initial decisions:
- Single-writer/multiple-reader software contract, synchronized handoff. No
  atomics or simultaneous conflicting writers. No OS isolation claim in SE.
- Contents live in the existing LLC CacheMemory arrays, never a shadow backing.
- Separate physical slot address from coherent source/destination addresses.
- Direct initialization by stores is required; no mandatory coherent zero copy.
- Existing bounded request/response resources remain in force. No free ports.
- Record unsupported cases and incomplete gates explicitly, rather than label
  compilation or matching checksums as correctness validation.

Status: source-path audit in progress; no new simulator or accepted result yet.

## First implementation, before compilation

- Added an experimental address selector (bit59), eight bank IDs, and existing
  3-bit way/10-bit set encoding. First ABI supports up to eight eligible ways
  per bank, at least two coherent ways, aligned accesses up to 256B. Broader
  geometry/unaligned accesses are not claimed. Local dependency comparisons
  now distinguish shared-bank identities from private slots.
- Native SPMCP obtains fresh data through the existing private source path;
  its final install is routed to the selected LLC bank, with completion only
  after remote installation. No private SPM slot is consumed for the transfer.
- Pure loads/stores route through the existing finite request/response NoC.
  First stores lazily claim slots and initialize their data in place. LLC slot
  claims use ordinary victim eviction, not silent metadata deletion/migration.
- Native SPMWB reads a shared slot, then follows the existing coherent
  overwrite path. Shared values remain snapshots, not coherent aliases.
- New 128B/256B payload classes charge vector transfers on Garnet correctly.
- Optional common bank-service model: one 64B/cycle service port, with wide
  SPM requests occupying multiple cycles. It conservatively charges ordinary
  controller messages too; it is an explicit model assumption, not RTL
  calibration. Enable identically in all new matched performance arms.
- SLICC generation passes. Functional tests and full compilation are pending.

## Generated-resource audit repair

The first candidate added a conditional enqueue inside existing private-copy
installation actions. SLICC checks such output resources unconditionally, so
this could throttle private SPM even with L3 disabled. Stopped only this new
build/its waiting runner (no simulations had launched); preserved build_v1 and
experiment_v1 as superseded. Restored original private install/ACK actions
byte-for-byte and added separate shared-source events/actions.

Both copy installation and writeback forwarding now consume their incoming
response into an existing, finite private-controller TBE before trying to issue
the next network request. A one-per-TBE timer schedules issue. This prevents a
full outgoing request FIFO from pinning a response at the input head. It adds
no unbounded payload queue and no accumulator storage. CPU completion remains
at final destination ACK. Candidate build_v2/experiment_v2 will repeat gates.

Guest_v2 includes eight-core sharing/RAW/masked-store/dirty-source/writeback/
release tests, dependent-load latency probes, and a minimally adapted FA
output-residency driver. FA retains private-L2 V staging; each core's FP32 O
region moves to its designated shared bank. This is a functionality transfer,
not a tuned L3 FA implementation, and no reduction arithmetic is offloaded.

## Host gates and queue preparation

Six host tests pass: namespace injectivity/capacity, unchanged private actions,
finite response capture, full vector network sizing, physical backing and
normal coherence-eviction routing. These are source/ABI checks, not runtime
proof. Build_v2 is compiling with the repaired source.

Before any simulator launch, the waiting experiment_v2 coordinator was replaced
by experiment_v3 to include checks that ordinary dirty pressure data survive
slot claims. Guest_v3 preserves prior guest attempts. The configuration audit
also explicitly permits the already-existing, disabled reduction parameters
absent from the older control configuration; it asserts reduction is OFF.
Authoritative new queue: experiment_v3/status.json. Two extra simulators max.

## Matched timing and final prelaunch queue

Experiment_v3 was superseded before simulation by experiment_v4 (PID344220).
Shared access latency is 35 cycles, matching the ordinary LLC micro-timing
profile, rather than assuming a faster SRAM access. A 70-cycle dependent-load
probe checks that the timing parameter actually affects observed latency.
Guest_v3 and build_v2 remain current. Full compilation passed; isolated linking
is underway. No runtime results accepted yet.

## Build complete; configuration packaging repaired

Build_v2 linked successfully, SHA256
`4d5abbd220a8179f4577fae5c751d9717c305a3511fe7cde717f91dc9d754ec8`.
Experiment_v4 failed before simulation because the build's generic frozen
configs lacked KVFixedMesh. The original control's actual experiment configs
contain it. Experiment_v5 now snapshots that complete configuration tree and
adds the L3 wrapper, matching the existing reduction runner's packaging. No
simulator source, topology, guest or hardware parameter changed for this repair.
The original failed launch is preserved. Six host tests continue to pass.

## Default-off regression gates passed

Experiment_v5 accepted both original-binary controls: cache121.1408us and
private-SPM116.3336us. Every prior finite numeric non-host statistic matches
exactly, full independent numerical checks pass, and output bits match. This
establishes default-off regression equivalence for these cases, not correctness
of the new shared path. Eight-core normal and two-entry-queue shared tests are
now active; no shared result has been accepted yet.

## Instruction-prefetch stale-fill regression

The normal shared smoke failed at tick193260400: physical L1 controller3,
address0xcc0, PF_Inst_IS received Data_Stale. Existing data prefetch PF_IS and
promoted instruction demand Inst_IS already handle this response, but the
unpromoted instruction-prefetch transition was missing. Added the matching
instruction-array cleanup without demand completion or retaining stale data.
Build_v3/experiment_v6 repeat all gates. Original failure and narrow old-binary
replay retained in diagnosis/prefetch_stale_v2. No timeout/resource enlargement,
prefetch disabling, or numerical tolerance change. Unrelated jobs untouched.

Old-binary trace confirms core3 instruction prefetch at192435200, private
request at192436000, LLC eviction/recalls, private IS_I->I stale fill at193256000,
and missing L1 instruction-prefetch handling at193260400. Trace is about16MiB,
not an unbounded whole-run log. The repaired build must demonstrate progress
and exact-value preservation; source reasoning alone is not acceptance.

## Speculative unclaimed-read semantics

V3 default controls again match exactly. Shared test then aborted on unclaimed
bank3/row16. Narrow trace diagnosis/copy_raw_v3 shows core2 SN433041 atPC400d18
(the cross-core read loop) speculatively reading bank3 row16, beyond its16-row
loop. The actual later copy-dependent load SN433087 reads its own bank2 and
correctly RAW-blocks on copies. Thus this was not broken copy completion.

Shared validation incorrectly rejected any unclaimed slot; private readSPMData
already returns zero without exposing coherent occupants. V4 retains geometry
checks, uses that existing zero semantics, and still requires stores/installs
to claim slots. Guest_v4 explicitly checks an unclaimed row before use. New
build_v4/experiment_v7 repeat all gates; no shared performance result yet.

## Shared runtime correctness gates passed

Build_v4 SHAba6e70a8dd0e983e2363df4c15b02a6c959534c7a8e3ac6fc3eb79e3b69a0c81.
Default-off controls again exactly match121.1408/116.3336us and all prior
numeric nonhost counters/output bits. Both eight-core directed tests PASS zero
errors, four rounds, including two-entry LLC queues. Checks cover cross-core
reads, masked writes, immediate RAW, dirty remote source copies, coherent WB
visibility without a post-WB fence, release/reuse, unclaimed zero reads, and
ordinary dirty data preservation across claims. Directed ROI101.0952us normal,
102.8052us small queues: these are tests, NOT FA speedups. Latency probes now
active, then checked T1024/T4096 triples. Eight host tests pass.

## Latency sanity accepted; FA launched

All six dependent-load probes pass for coherent/private/shared modes and one
or eight active workers. One-worker256-load ROI: coherent0.8456us, private
SPM1.3336us, shared9.7652us. Eight-worker shared10.2620us. Shared service delay
35->70cycles changes one-worker ROI9.7652->13.3480us (+3.5828us), consistent
with256*35*0.4ns=3.584us. Thus the configured latency is exposed by a serial
dependency chain; this is not an FA slowdown attribution or calibrated silicon
latency. T1024 matched FA controls now running, shared-output next, T4096 queued.

## Automatic queue journal: experiment_v7


2026-09-28T08:44:15+00:00

- Queue stage: `fa_1024`.

2026-09-28T08:45:00+00:00

- Queue stage: `fa_4096`.
- Accepted `t1024_l3_output`: ROI 173.4896 us; all configured numerical/config/resource checks passed.

2026-09-28T08:54:46+00:00

- Accepted `t4096_private_spm`: ROI 1167.5820 us; all configured numerical/config/resource checks passed.

2026-09-28T08:55:01+00:00

- Accepted `t4096_cache`: ROI 1194.3868 us; all configured numerical/config/resource checks passed.

2026-09-28T08:56:16+00:00

- Queue stage: `failed`.

Queue stopped on a failed gate; see status.json and failed.json. Failed timing is not an accepted result.
