# L3 accumulator lifecycle experiment — 2026-09-28

Related ledger L3-02/L3-03. Existing pipeline+FP16 reduction improves T1024
reduction latency4.118% but remains22.123% slower than privateSPM. User authorized
first-update fusion, then near-L3 normalization/conversion/writeback/release,
with fixed query mapping and matched end-to-end comparisons. This differs from
changing cache sizes, queue budgets, or using extra accumulator SRAM.

## Design decisions

- Bit56 on a typed reduction initializes the row from explicit +0 and the first
  contribution. Normal recall/eviction precedes slot claim. No zero upload or
  SRAM accumulator read. FP32 mul/add semantics preserved, including first-update
  rounding; still charges conversion/ALU time and final256B SRAM write.
- Bit57 on an existing SPMWB instruction requests finalization of two FP32
  slots (32 elements,128B) into one coherent64B FP16 line. Scalar FP32 reciprocal
  travels as metadata. Two commands finalize a64-element output row. The regular
  SQ/DTLB performs normal destination translation; no functional-memory writes,
  host pointer translation oracle, or CPU accumulator load. The existing64B
  coherent destination RAW/WAW and barrier completion tracking remains active.
  SPM source shadow/hazard footprint extends to two slots per command.
- Bank finalization shares the four-entry sink,16-lane ALU and shared64B/cycle
  SRAM port. Read35cy, multiply+narrow8cy, original35cy response, and2cy slot
  reclamation cost. Final FP16 data stays in finite protocol buffers; slots can
  be released after the snapshot is secured. Local dependency/barrier tracking
  persists through coherent writeback acknowledgment. No extra accumulator SRAM.
- IMPORTANT limitation: existing SPMWB writes through the LLC to memory and
  waits for the memory-directory acknowledgment. This policy remains unchanged.
  Finalization therefore is not a pure arithmetic ablation versus CPU cache
  stores. Measure its DRAM/output-write traffic; do not interpret a slowdown as
  proof that near-L3 finalization itself is unattractive. The finished64B output
  travels L3->private controller->coherent home, not through CPU registers; this
  is a correct reuse of existing routing, not the minimum-hop possible design.

All timings are explicit model assumptions, not calibrated RTL. Previous frozen
builds, source snapshots and results preserved. Existing pipeline and unrelated
queues continue untouched. New root results/cacheflex_l3_spm/lifecycle_20260928.

FA geometry is inherited unchanged from pipeline experiment_v2: single head,
d64,8cores,VL16,query partition. FP16 operands/PV contributions; FP32 running
softmax/accumulator; FP16 stored output.64KiB L1I/D,512KiB private L2/core,
4MiB shared L3 (8x512KiB banks),128bit fabric, same finite resource profile.
T1024 Bc128 and T4096 Bc512; complete inherited configuration diffs audited.
Previous full T4096 factorial has finished: cache1200.6220/private1170.4900,
FP32base1177.7440/pipeline1180.3720/FP16-only1159.5416/combined1163.4248us.
Thus fix pipelinedFP16 atT1024 and serializedFP16 atT4096 for these lifecycle
ablations. This is a fixed mapping/engine test, not a fresh global tuning claim.

## Gates and comparisons planned

Exact default controls on the old ELF; initialization/finalization directed
tests (dependent reads, dirty output, cross-core visibility, slot reuse, finite
one-entry stress). Then same-new-ELF cache/private/base/init/final/both atT1024
andT4096, with full independent oracle, exact output bits, command/slot/writeback
counts, bounds and configuration audit. Maximum two new simulators. Fail closed.
Primary metric end-to-end ROI; queue retry or command cycles are not runtime
fractions. Expected output-store DRAM policy is reported explicitly.

## Validation log

- build_v1 stopped at a SLICC cache-entry overload error; corrected with the
  existing AbstractCacheEntry setter form. build_v2 successfully linked.
- All31 source guards pass. experiment_v1 exact old-ELF cache/private/reduction
  controls pass ALL finite nonhost stats and full independent arithmetic/output
  equality:119.6668/115.5796/141.1488us. Initialization directed test passes
  768updates,192first-update claims, three reuse epochs, zero errors.
- Finalization directed arithmetic failed; peers and FA queue stopped. No
  finalization counted despite384ordinary SPMWB completions: Request's shared
  slot metadata canonicalization discarded bit57 and reciprocal bits55:24.
  Repair preserves these only for marked shared SPMWB; downstream Ruby still
  uses canonical slot addresses. Failed logs/stats/build retained. build_v3
  rebuild; all gates must repeat before FA. No timing/capacity changes.
- Automated launch: CACHEFLEX_L3_LIFECYCLE_BUILD=build_v3
  CACHEFLEX_L3_LIFECYCLE_EXPERIMENT=experiment_v2 python3
  scripts/launch_l3_lifecycle.py. It waits for a successful frozen build, refuses
  existing output directories, and starts run_l3_lifecycle.py. Companion
  analyze_l3_lifecycle.py --watch maintains accepted-only EFFECTS.md/effects.json.
- build_v3 COMPLETE SHA f9ecddd34dada060e87f9bfdcd9884dd98b8a225c19400e883dc4b636cd92b53.
  experiment_v2 automatically launched exact cache/private controls345362/345575;
  remaining exact/directed/performance checks queued. Guest unchanged.
- experiment_v2 exact controls again pass, but finalization now reaches a new
  ABI assertion that incorrectly expected a64B CPU packet. Existing SPMWB ISA
  carries an8B slot descriptor; the controller always writes64B. Corrected the
  assertion to descriptor size8 and coherent destination line alignment. Full
  64B coherent hazard coverage was already present. No queue/timing change.
  build_v4/experiment_v3 repeat gates; v2 failure evidence retained.
- build_v4 COMPLETE SHA8361202f335e75d9d6b8e8883525fcdf4dfd55d73858b1fb20c97b007445c7ca.
  experiment_v3 starts from all default/directed gates;31source guards pass.
- v3 directed finalization exposes cross-SQ RAW ordering: core7 WB SN3804 first
  attempts37199600, before older SPM reduction SN3464's send attempt37218400.
  Bank finalization releases row0 at37279600; another reduction arrives after
  release and asserts at37424800. Complete bounded trace (stopped just before
  panic): diagnosis/ordering_v4_complete/trace.gz. Unbounded panic replay's gzip
  is truncated by abort; complete replay is the authoritative evidence.
- Fix in LSQUnit::writebackStores: bridge source reads check existing SPM SQ
  for older overlapping stores, plus existing bridge shadows for pending copies.
  Once captured, posted reductions remain protected by Ruby reductionPending.
  No new buffer, global fence, or relaxed numeric check. Applies to ordinary
  SPMWB source RAW correctness too; new finalization additionally protects2slots.
  build_v5 SHA999611a66a64b764cb4d3db076fc049ef4c38664e03e6e765ea43c27775d9691.
  experiment_v4 repeats all gates; older failed runs preserved.
- experiment_v4 exact controls PASS (119.6668/115.5796/141.1488us, all finite
  nonhost stats and full independent output equality). Initialization PASS
  768commands/192initializations, finalization PASS768commands/384writes; zero
  arithmetic errors, three reuse epochs, dirty/shared output and unfenced
  dependent reads. Both/bounded-one-entry/sync/serial gates pending at entry.
- ALL9 runtime gates now PASS (3exact controls+6directed), gates_passed.json
  exists. One-entry, synchronous and serialized combined modes all zeroerrors.
  FA same-new-ELF T1024 six-arm screen active, followed automatically by T4096.
  Coordinator2056439; max2sim. Every FA target uses full independent oracle,
  exact prior output bits, unchanged copy count, configuration and bound checks.
- ALL6 T1024 FA cases accepted: cache121.0324/private115.8060/base140.9140/
  init128.8536/final168.8404/both156.5028us. Init removes8.559% of reduction ROI;
  finalization adds19.818%; both adds11.063%. Best remains privateSPM. Every
  output element independently checked (65536), exact prior output bits too.
  Init1024claims and final2048writes exact. Finalization adds131072DRAM bytes
  (257536->388608) through unchanged write-through policy; network flits fall
  slightly (480894->475673) despite slower ROI. MeanWB281Rubycycles overlaps
  other work; this is not an isolated DRAM-latency attribution.
  T4096 cache/private pair now active; remaining four arms automatically queued.
