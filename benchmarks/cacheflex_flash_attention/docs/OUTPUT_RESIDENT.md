# Persistent FP32 output, coherent Q/K/V

Current experiment: `results/cacheflex_flash_attention/output_resident_v10_20260924`.
Guests, repaired model and initial failure: `output_resident_20260924`.
Runner: `scripts/resume_output_resident.py` (reuses validation/command helpers
from `scripts/run_fa_output_resident.py`); status.json is authoritative.

Single head, T512 numerical gate then T4096 target, eight query-partitioned
cores, d64/VL16. Br16/Bc128/G4: 64 query rows/group. Each worker keeps its
64x64 FP32 running output (16KiB) in two existing private-L2 ways throughout
all K/V tiles. Q/K/V, probability tiles and softmax m/l remain coherent.
QK and PV retain the existing FP16 microkernels; rescale then accumulation
remain separately rounded FP32 operations. Final normalization writes FP16
output to coherent memory once. No fusion, DMA, multicast or new data SRAM.

The simulator inherits the IFETCH-on `ifetch_edge_20260923` model with a scoped
SPM-store acknowledgment correctness repair. Actual L1I/D=64KiB, private L2=64KiB, shared LLC=4MiB.
Normal miss pools, ports, controller timing and network remain unchanged.

## Layout and costs

Rows r=0..63 occupy sets 2r and 2r+1 in ways 0/1. Each 128B half-row spans
two ways at one set. Native full-width FP32 SPM accesses would span four ways;
the existing dtype 0xb (`spm.ld1ad/st1ad`) transfers 32 32-bit words into/from
64-bit register lanes. UZP assembles the load; ZIP prepares the store. This
is bit-preserving transport, not FP32-to-FP16 conversion. Both instruction
and port costs are measured. It does not rely on masked-off bytes suppressing
requests or allocate inaccessible SPM slots in ways 2/3.

SPMST to an unclaimed slot does not allocate it in this implementation.
Each core therefore copies its existing pre-zeroed 16KiB Oacc allocation to
claim 256 lines once, within the ROI. A barrier drains these initial copies.
Thereafter groups initialize the same slots using SPM stores, never restaging
K/V. The original Oacc allocation is retained solely for bootstrap, not as
extra SPM backing. The end of the worker releases all 256 slots, also timed.
Total copies: 2048, independent of T. Normal stores publish the normalized
FP16 output; no SPMWB of unnormalized FP32 scratch is necessary.

## Comparisons and acceptance

- New coherent counterpart from the same transformed source, Br16/G4.
- Revalidated historical cache winner Br32/G2.
- Revalidated historical V-SPM Br16/G4, also two ways.

All use the same simulated hardware. Historical control reuse does not imply
new coherent code/layout is identical: report the new matched control as well.
All outputs must pass the existing independent tile-aware numerical oracle,
match historical output bytes, and match the guest's printed hash. Full config
comparison allows only executable and the declared cache tiling argument.
Check exact copy count, non-speculative SPM store count, releases, and zero
global sequencer guard rejects. No automatic tuning or hardware expansion.

The initial T512 SPM run exposed a wide-store bug: SLICC had performed the
128-byte private-L2 store, but generic Sequencer::hitCallback attempted to
write the payload again into its 64-byte acknowledgment DataBlock. The repair
skips this redundant write for SPMST. It also reissues followers after an SPM
store acknowledgment: the ack is not an L0 cache line that can service queued
loads/stores. All prior resources and service latencies are unchanged. V2
requires old cache and V-SPM runs at both sizes to match every finite numeric
non-host stat, in addition to arithmetic/config checks, before accepting the
new comparison. Initial failure logs, sources and simulator remain preserved.

V2's old T512 controls passed exact counter equivalence, but the new guest then
exposed a second wide-store defect. RubyRequest::WTData defaulted to 64 bytes;
Sequencer copied 128 bytes into it through an unchecked DataBlock pointer copy.
V3 uses build_model_v2, which sizes this existing message payload to the vector
request and adds bounds assertions. This is not additional SPM/cache capacity.
The old V2 guest and all failure evidence are retained. V3 first runs a 64-round
4096-bit-pattern wide-store/load regression, then repeats the equivalence/math
gates before timing. A stopped memory-checker diagnostic is NOT a memcheck pass.

The V3 regression then exposed missing timed-store ordering: SPMST could pass
an older pending slot-claim copy; DSB tracked only pending writeback bridges.
Also, pure SPM loads skipped older-store RAW checks under an incorrect
assumption of synchronous L2 writes. V5/build_model_v4 adds way/slot-overlap
waits for these dependencies (without store forwarding) and marks/drains both
committed bridge directions for barriers. No capacity/latency knob changed.
V4 was stopped before any runs to finish the retirement bookkeeping. The
regression and numerical/counter gates must pass on the final model.

V5 still failed: the plaintext `roundtrip_order_plain` trace isolated the
immediate cause as incorrect Sequencer coalescing, not the ordering omissions.
Seven copies followed pending coherent initialization stores; Ruby acknowledged
them without ever installing their SPM destinations. V6/build_model_v5 reissues
SPM followers through the controller rather than completing them against an
ordinary store response. The previous ordering changes are separate repairs.
All runtime gates remain required; no new FA performance result is accepted yet.

V6 then completed the roundtrip but failed 2048/4096 bit comparisons.
`roundtrip_v2` shows lanes 16..31 repeat lanes 0..15: `writeSPMData` reused
the first payload slice for every way. V7/build_model_v6 advances its payload
offset while keeping the destination offset local to each way. SPM bridge
acknowledgments also reissue followers, just like SPM stores. Runtime validation
is pending; these repairs do not add storage or change timing parameters.

V7's 64-round regression now passes all 4096 bit-pattern checks. Existing-kernel
equivalence and full FA arithmetic gates are running before T4096 timing.

V7 rechecks: T512 cache is exact in all finite non-host stats. V-SPM retains
149.386us ROI, all instruction counts and all memory counters, but core6 has
nine more awake-empty cycles and nine fewer sleeping cycles (26 affected
idle/distribution counters). V8 permits only that exact recorded delta from
`preserved/t512_idle_counter_delta.json`; all other differences still fail.
Arithmetic/configuration checks are unchanged. Both controls run again under
the same model as the output-resident cases; old results remain preserved.

V8 T512 gates passed: output-SPM 172.2084us, same-source cache 172.7592us;
both match the independent reference and historical output bytes, with exact
copy/store/release counts and accepted config audits. Both are slower than the
retained cache winner (151.0676us) and V-SPM (149.3860us) at this small size.
These are validation cases, not evidence of target T4096 benefit. T4096 controls
are active, with the output-resident/matched-cache pair queued behind them.

V8 stopped at the old-counter equality gate on T4096, not a numerical error.
The completed V-SPM control passes all 262,144 checks, unchanged output bytes,
configuration, resource bounds, committed work and copy count. Runtime changes
by -0.006950%. A bounded old/new trace audit identifies the first divergence:
the repaired core drains a committed copy before handling a faulting regular
load, whereas the old model handles the fault while that copy is outstanding.
The exact evidence is `output_resident_20260924/control_audit/REPORT.md`.
Do not interpret the 479-cycle local wait as an end-to-end penalty; total ROI
is slightly lower, and later scheduling/arbitration differ.

V9 keeps the repaired behavior and reuses unchanged accepted T512 cases.
It reruns T4096 cache and V-SPM controls; cache must match historical counters,
V-SPM must match ALL finite numeric non-host counters of the first run on the
same repaired model. Full numerical/configuration/resource checks remain.
Then output-resident and matched-cache T4096 run against those fresh controls.
No simulator, guest, capacities, timings or arithmetic tolerances changed.

V9 SPM repeat is EXACT in every finite numeric non-host counter in both stats
blocks versus V8. A second gate error compared historical committed work after
the ROI as well: FA work matches, but shutdown/export has 56 extra instructions
and 60 extra operations. V10 restricts historical-work comparison to the timed
ROI, while retaining full all-block same-model reproducibility. The completed
SPM control (5715.0284us) is reused. V10 runs the interrupted cache control and
actual output-resident T4096 in parallel, then the matched coherent kernel.

Host tests: `python3 -m unittest discover -s scripts -p test_fa_output_resident.py -v`.
The result tests output placement under current hardware; it is not proof of
new architectural novelty, optimal tuning, or that output traffic lies on the
critical path. The MICRO repository already contains an output-resident variant.
