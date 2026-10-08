# CacheFlex experiment ledger

Established 2026-09-27. This is the decision-facing memory of completed experiments, including negative results. Read it before recommending or launching another experiment. Raw evidence is preserved; nothing was rerun to create this ledger.

## How to use this record

1. Search the mechanism, workload and synonyms here and in [the evidence index](INVENTORY.md). `python3 scripts/index_experiment_history.py --query forwarding` searches individual saved evidence paths too.
2. Cite the relevant experiment IDs when proposing a direction. State **what was tried, what happened, and what materially differs in the proposed test**. A new model/configuration justifies a transfer/revalidation, not a claim of a new idea.
3. Prefer an explicit untested interaction or a discriminating causal test to another broad parameter sweep. Negative results are local to the tested workload/model, not universal impossibility proofs.
4. Record new accepted results here, including null/regressive results. Record accepted subsets of stopped queues separately from unfinished targets. Update the evidence index after completion.
5. End-to-end ROI delta is the primary performance outcome. Summed stalls, request lifetime percentages, copy counts and traffic changes do **not** give additive removable-runtime fractions.

### Evidence grades

- **N**: saved numerical/reference and relevant configuration checks support the listed measurements. This does not certify the simulator or paper readiness.
- **D**: useful controlled diagnostic, regression, trace or timing observation. Scope may be narrower than a full correct workload.
- **H**: historical/preliminary evidence with weaker checks, confounding configuration, or subsequently identified arithmetic defects. Useful for avoiding repeated mistakes; not validated FA/model speedup.
- **X/P**: failed, cancelled or pending portion. No performance conclusion from that portion; completed accepted subsets can still be N/D.

An `accepted.json` filename is not sufficient to override a subsequently discovered defect. The machine index records marker presence, not independent revalidation.

### Coverage and provenance

This ledger groups experiments by scientific question rather than giving each smoke test a new hypothesis. The accompanying [runs.csv](runs.csv) and [inventory.json](inventory.json) index individual result directories and available acceptance records; [INVENTORY.md](INVENTORY.md) links saved reports and summaries. [historical_context.json](historical_context.json) preserves the previous AGENTS.md experiment notes, including superseded entries, as evidence rather than current status.

The initial scan covers all `results/` areas, excluding copied source repositories, toolchains, build trees, configs and reference-data trees. It found 569 study roots and 4,075 run/diagnostic directories, **not 4,075 independently validated performance experiments**. Raw-only cases without sufficient surviving validation remain H/unverified rather than silently promoted. First stats dump values in the index are not automatically ROI: some old drivers use per-panel epochs or whole-process dumps. Reports/accepted epoch records take precedence.

## Quick mechanism lookup — do not rediscover these

| Proposed idea | Prior evidence | Current conclusion |
|---|---|---|
| Stage K+V instead of V only | FA-03, FA-06, LEG-02 | Tried extensively; sometimes improves K-only but did not beat the historical V-only winner. Latest finite T16K kernel has not transferred it. |
| Pin K or enlarge query groups to reduce copying | FA-03, FA-09, FA-29 | Copy savings can be offset by coherent set pressure/output footprint. |
| Stage private Q | FA-10 | Lower DRAM but worse PV/K delivery and more Q adapter instructions. |
| Keep output accumulators in SPM | FA-25–27 | Full FA implemented and checked; slower than best cache. Forwarding zero gain; early-address improvement 0.39%. |
| Directly generate packed probabilities / SPM-native producer layout | FA-06, FA-29 | Existing software fusion options are not direct softmax-to-packed-P fusion. No accepted full-size implementation of this exact mechanism found in this audit. |
| LLC retention for silent copies | FA-08 | Large traffic reduction, small latency gain in tested FA cases. Already implemented/tested. |
| More sequencer entries / reserve requests | FA-12, FA-14, MOD-01–03 | Larger shared budget did not reliably help; newer finite model uses explicit resources and a nonbinding guard. |
| Async bounded staging / double buffering | FA-14 | Tested; small cap/retry gain, double buffering substantially worse in scaled-L2 FA. |
| Faster SPMCP / SPMLD | FA-04, FA-15–24 | Many tests; usually marginal. Nearly ideal repeated staging also tested. |
| FIFO bypass / demand priority | FA-17–19 | Actual local wait removed in one test; full-runtime gain small, some bypass variants regress. |
| Faster fetch / wider network / more miss credits | FA-15, FA-22–24 | Real sensitivity exists, but these also help coherent cache; compare matched platforms. |
| Multicast / batch / per-core DMA | GEMM-05, LEG-01–04 | Already implemented and tested; historical FA arithmetic caveat applies. |
| Dedicated/separate SPM backing | FA-04, FA-12 | Diagnostic only: adds storage/removes displacement, not an area-matched CacheFlex improvement. |
| SwiGLU / grouped projections | AI-03–08 | Promising accepted subsets; warm and first-call results must remain separate. |

## Model boundaries

Do not pool ratios across these profiles:

- **Legacy FA through early Sep11:** dense FP16 conversion defect; matching hashes did not prove attention arithmetic. Some controlled frontend/regression findings remain useful.
- **Corrected multihead Sep11–15:** typically Hq8/Hkv1, d64, eight cores, VL16, 512 KiB private L2. Consult each saved config for aggregate LLC; some older runners actually had 32 MiB, not 4 MiB.
- **Scaled single-head Sep15–24:** typically 64 KiB private L2/core, 4 MiB total LLC, T4096 or T16K, fixed older query kernels. Resource and timing revisions within this period matter.
- **MICRO reconstruction:** Hq32/Hkv8, one core, 512 KiB L2; Classic/Ruby and original/common harness ratios are separate experiments. No original paper-winning result manifest was available.
- **KV/common-driver Sep25:** one head, 512 KiB private L2, 4 MiB LLC, corrected timing, but pre-Sep26 finite-network profile.
- **Finite Sep26:** bounded LQ/SQ/bridge shadows, finite buffers/TBEs, MICRO miss/WB pools, request-VC partition using the same total VC resources; nonbinding sequencer guard. Current analytical winners belong here.
- **SwiGLU model-shaped pilot:** separate frozen older simulator, 512 KiB L2, 4 MiB LLC, sequencer16 and two SPM load ports; FP32 GEMM accumulation. Do not causally compare its ratio to newer FA hardware.

## Corrected FA: placement, scheduling and kernel optimization

### FA-01 — Packing, launch-state and instruction placement

**D/N, Sep11.** Canonical argv made byte-identical archived/rebuilt SPM guests agree at 936.792 us. Fixed-region placement tests demonstrated a real fetch-boundary cost: corrected SPM QK 871.3032 → 927.5712 us and cache PV 848.9056 → 906.5488 us when crossing a 64B boundary. Old-arithmetic packing/alignment tests reduced an apparent 88.4064 us gap to 4.8844 us, but that historical percentage is not a corrected-FA result. Corrected Q/P packers reduce cache 866.4424 → 842.5892 us and SPM 881.4056 → 870.0764 us.

**Takeaway:** control argv, code alignment and emitted instructions before calling a driver/layout effect architectural. Software packing benefits both paths. Sources: [corrected audit](../../results/cacheflex_flash_attention/correct_fa_20260911/REPORT.md), [PV evidence](../../results/cacheflex_flash_attention/pv_alignment_20260911/FINDINGS.md), [launch controls](../../results/cacheflex_flash_attention/driver_launch_20260911/summary.json).

### FA-02 — Matched query grouping

**N, Sep11; T1024, Hq8/Hkv1, Br64/Bc128.** Groups 1/2/4/8: cache 854.4056/854.3252/842.5892/861.8912 us; rolling K-SPM 925.2292/895.1668/870.0764/879.5432 us. Both select group4; SPM is 3.26% slower.

**Takeaway:** reuse is not monotonic with group size; live Q/O storage grows too. Do not automatically reuse the largest group. [Source](../../results/cacheflex_flash_attention/correct_fa_20260911/REPORT.md).

### FA-03 — Rolling versus persistent K; qualified cache-residency tracing

**N/D, Sep11.** Group4 rolling K: 32 KiB/core, 65,536 copies, 870.0764 us. Pinned K: 128 KiB/core, 16,384 copies, 868.0868 us. With extra separate backing these become 845.7512 and 828.1972 us. Nearly 99% of additional qualified O/V/packed-Q removals occur in the claimed set range. Full-size nonperiodic check: cache 841.2800, pinned SPM 874.3404, separate-backing pinned SPM 825.4496 us; all outputs checked.

**Takeaway:** staging less can help, but set-local displacement cancels it. Four-cell interaction is stronger evidence than copy counts. Extra backing is not a proposed area-matched solution. [Audit](../../results/cacheflex_flash_attention/correct_fa_20260911/REPORT.md), [buffer evidence](../../results/cacheflex_flash_attention/correct_trace_20260911/BUFFER_REPORT.md).

### FA-04 — Faster copy/load versus capacity-preserving diagnostic

**N, Sep11; same corrected executable.** Default 870.0764 us; faster-copy/more-copy-port bundle 870.3468; faster-SPMLD/more-load-port bundle 869.3068; separate backing 845.7512. Removing displacement recovers 24.3252 us, 88.5% of this specific cache/SPM gap, while adding storage.

**Takeaway:** that K-staging configuration was insensitive to tested service acceleration and sensitive to capacity/claim effects. Not a universal FA limit or isolated per-port effect. [Source](../../results/cacheflex_flash_attention/correct_fa_20260911/REPORT.md).

### FA-05 — Resident QK and unresolved-address retry

**D, Sep11.** Old resident-SPM guest deadlocked after copies completed because deferred loads never reentered issue. Deduplicated retry repair completes cache/SPM resident QK at 170.2000/168.5792 us, with no copies inside ROI and analytic checks passing.

**Takeaway:** this was simulator liveness, not a valid performance limitation. Isolated resident GEMM is not full FA. [Regression](../../results/cacheflex_flash_attention/spm_retry_check_20260911/REPORT.md), [failure](../../results/cacheflex_flash_attention/resident_stall_origin_20260911/DIAGNOSIS.md).

### FA-06 — Corrected staged search, including K/V/K+V, buffering and placement

**N with periodic-screen limitation, Sep12–13; T1024/Hq8/Hkv1.** Saved periodic candidates include 118 K, 124 V, 30 K+V and 116 cache cases. Best screened K+V 719.7316 us versus V-only 673.3564 us; independently selected settings, not a pure placement delta. Matched K+V sometimes beats K-only (Br32/Bc512/G8: 761.056 → 719.7316 us), sometimes regresses. Searched tiles/grouping, single/double buffering, pinning, offsets, QK width/pipeline/unroll/load order, rescale fusion and phase ordering. Full-size rich-input finalists checked separately; later accepted SPM finalist 672.5420 us.

**Takeaway:** K+V is already tried, not a new direction. V-only won this bounded search. Not every implemented dimension is a novel mechanism; direct softmax-to-packed-P fusion remained unimplemented. [All candidates/finalists](../../results/cacheflex_flash_attention/best_corrected_20260912/summary.json), [plan](../../results/cacheflex_flash_attention/best_corrected_20260912/plan.json).

### FA-07 — Optuna and full-width PV / stride hoisting

**N, Sep12–13.** Optuna best cache/SPM 684.2544/689.2604 us (0.992737x), whereas staged SPM finalist was 672.5420 us. Three repeats were identical, not independent stochastic confidence. Full-width PV improves cache 684.2544 → 656.6536 and SPM 672.5420 → 651.2724 us. Constant-stride replacement alone gives 673.0916 us; with full-width PV 655.5920, both worse than corresponding unpatched SPM.

**Takeaway:** optimizing useful vector work helped both more than removing the flagged stride reload. A PC hotspot is not guaranteed removable latency; Optuna did not universally beat staged search. [Optuna](../../results/cacheflex_flash_attention/optuna_20260912/summary.json), [PV/stride](../../results/cacheflex_flash_attention/pv_improvements_20260913/summary.json), [winner attribution](../../results/cacheflex_flash_attention/winner_analysis_20260913/REPORT.md).

### FA-08 — Larger T and LLC retention of silent fetches

**N, Sep14–17.** Multihead T16K cache/SPM-no-retention/SPM-retention: 143.0413912/145.9283456/143.0443268 ms. SPM DRAM reads fall 663,664,640 → 24,772,480 B, but latency improves only ~1.98%, reaching parity. Single-head scaled-L2 T16K: cache 67.232142, SPM off 62.0861492, on 61.3072440 ms; DRAM reads 551,536,192 → 180,179,648 B.

**Takeaway:** no-fill repeated silent fetches are real; retention helps traffic much more than runtime here. Data can be LLC-resident and still require repeated on-chip SPM copies. [Multihead](../../results/cacheflex_flash_attention/llc_retain_20260915/status.json), [single-head](../../results/cacheflex_flash_attention/single_head_retention_20260917/summary.json), [DRAM diagnosis](../../results/cacheflex_flash_attention/long_context_pv_20260914/DRAM_DIAGNOSIS.md). T32K was cancelled: no complete T32K result.

### FA-09 — Analytical traffic model and larger query group

**N/D, Sep15.** Group512 multihead T16K cache/SPM 142.5078844/141.9928184 ms (1.003627x); copies 8,388,608, half the earlier group256 count. SPM still reads 312,088,256 B from DRAM versus cache 29,228,160 B. Analytical model predicts copy/footprint trends, not exact latency.

**Takeaway:** amortizing more copies did not produce substantial gains; retain actual measured misses/latency rather than treating logical traffic as DRAM. [Measured pairs](../../results/cacheflex_flash_attention/group512_t16k_20260915/summary.json), [model](../../results/cacheflex_flash_attention/analytical_model_20260915/REPORT.md).

### FA-10 — Persistent private Q versus streamed V

**N/D, Sep15–17; single-head T16K, scaled 64 KiB L2.** Cache/V-SPM/Q-SPM: 67.232142/62.086149/73.173701 ms. Q staging reduces DRAM by 79.49% versus V, but packets rise 40.27% and private-L2 misses 78.32%. Q load adapter uses six instructions instead of the coherent broadcast load, adding ~83.9M replacement instructions. Q used repaired protocol; older cache/V results were on the preserved earlier simulator.

**Takeaway:** private/long-lived data is not automatically the best SPM operand. This implementation loses the V-side advantage and pays adapter costs. Not an isolated proof that all Q staging is inferior. [Analysis](../../results/cacheflex_flash_attention/q_resident_analysis_20260917/REPORT.md).

### FA-11 — Recording overhead and clean controls

**N/D, Sep16–17.** Flight recorder off/on T512 produces identical 113.8272 us simulated timing; example host times 22.95/44.74 s. Repaired recording-off T16K cache/V-SPM reproduce 67.232142/62.0861492 ms and full numerical checks.

**Takeaway:** host duration is not simulated performance; bounded trace recording can nearly double host cost. Disable except for targeted diagnosis. [Recorder](../../results/ruby_flight_20260916/validation/passed.json), [controls](../../results/cacheflex_flash_attention/clean_controls_20260917/REPORT.md).

### FA-12 — Sequencer 16/32/64/128, stride, service and separate backing

**N/D accepted subsets, Sep17–20.** Single-head T16K SPM at 16/32/64/128 entries: 61.307244/64.579371/74.632957/76.527760 ms; cache at 32/64/128: 65.376946/67.397546/64.835340 ms. At16, stride patch 61.176752 ms; fast load 61.267932; fast copy 61.387088. Separate backing 45.289159 ms; backing+fast service 45.311517. Multihead long sweep/overnight chain was stopped on user request; completed small gates remain evidence, not complete targets.

**Takeaway:** raising admission does not monotonically help; can move pressure downstream. Extra backing is an optimistic non-area-matched diagnostic, not the desired mechanism. [Accepted single-head cases](../../results/cacheflex_flash_attention/single_head_diagnosis_skip_controls_20260918/REPORT.md), [overnight](../../results/cacheflex_flash_attention/overnight_diagnosis_20260918/REPORT.md).

### FA-13 — Admission/staging dataflow traces

**D, Sep17–21.** Selected multihead stride load waits111 cycles before Ruby admission with16 copies outstanding, then completes locally in ~1 cycle. Downstream trace: all16 blocking copies LLC hits; request injection and return transport hold them. Full retained T16K audit finds 99.8488% of16,777,216 copies served by LLC; 1 GiB copy payload.

**Takeaway:** real shared-admission interference and repeated on-chip movement exist, but selected worst-gap traces do not establish runtime fraction or expected DMA gain. [Admission](../../results/cacheflex_flash_attention/staging_trace_20260917/ANALYSIS.md), [dependencies](../../results/cacheflex_flash_attention/spm_dataflow_trace_20260918/DEPENDENCIES.md), [whole-ROI audit](../../results/cacheflex_flash_attention/staging_cost_audit_20260921/REPORT.md).

### FA-14 — Bounded async copy cap/retry isolation plus double buffering

**N, Sep21.** Scaled-L2 T4096 SPM control 4.512217 ms, cap12 4.475614, independent retry+cap 4.466819 (~1.01% reduction). Double-buffered JIT 5.922164, best shown chunked isolated 5.835972. T16K original SPM 61.307244 ms; double-buffer JIT 79.552281, chunked isolated 80.704708; matched cache 67.232142. Same64 KiB L2; extra buffer occupies more existing SPM slots, not new SRAM.

**Takeaway:** async bounded staging is already tried. Small admission improvement did not compensate for buffering footprint/driver costs. Do not generalize to a different K/O layout without stating the change. [Full table](../../results/cacheflex_flash_attention/async_staging_20260921/experiment/REPORT.md).

### FA-15 — Retention × network × prefetch × admission interactions

**N, Sep20–21.** Forty accepted cases in the capacity-preserving round. Single-head T16K network128→256 reduces SPM 61.307→47.011 ms (~23.32%) and cache 67.232→49.700 (~26.08%). Matched ratio falls1.097→1.057x. Tested restrained/no prefetch, retention off and combinations, admission8; no extra backing.

**Takeaway:** on-chip memory/network headroom is real but widening is not an SPM-specific gain. Never compare wide-SPM against narrow-cache as the architectural ratio. [Queue/results](../../results/cacheflex_flash_attention/contention_optimization_v2_20260920/status.json), [interpretation](../../results/cacheflex_flash_attention/staging_cost_audit_20260921/REPORT.md).

### FA-16 — Fine timestamps and all-copy aggregate

**D/N, Sep22.** One selected T512 copy takes711.25 cycles; not representative. Full T4096 covers524,288 LLC-hit copies, mean issue→install364.644659 cycles: ingress FIFO53.992%, network35.218%, local/controller6.065%, CPU3.060%, own forwarding resource wait1.666%. ACK additional7.875 cycles. Observer on/off and occupancy-integral checks match.

**Takeaway:** these partition request lifetimes, not FA runtime. All-copy coverage corrects overgeneralization from one WB-blocked request. [Aggregate](../../results/cacheflex_flash_attention/access_aggregate_20260922/experiment_v3/REPORT.md), [single copy](../../results/cacheflex_flash_attention/access_timing_20260922/experiment_v3/BREAKDOWN.md).

### FA-17 — Bounded ingress lookahead

**N, Sep22.** T4096 SPM W0/W4/W16: 4751.4864/4788.9660/4787.7472 us. W4/W16 worsen runtime ~0.79/0.76%; ready→successful-forward increases, despite earlier first probes. Probes consume existing controller transition budget.

**Takeaway:** an earlier attempt is not earlier service. This bypass implementation did not remove the suspected bottleneck, so its negative result does not prove all FIFO isolation useless. [Results](../../results/cacheflex_flash_attention/ingress_bypass_20260922/experiment/REPORT.md), [interpretation](../../results/cacheflex_flash_attention/ingress_bypass_20260922/INTERPRETATION.md).

### FA-18 — Credit ownership and copy-completion slack trace

**D, Sep22.** Sampled private-L2 miss pool20 is predominantly copy-owned; selected ordinary L1-hit driver load waits312 cycles behind copies. For seven sampled panels, all V copies finish9,045–12,731 cycles before first V load. Trace on/off counters match.

**Takeaway:** copy interference with unrelated work exists, but V itself has large completion slack. Accelerating fills need not accelerate consumers. [Evidence](../../results/cacheflex_flash_attention/ingress_trace_20260922/REPORT.md).

### FA-19 — Demand bypass around blocked copies

**N/D, Sep22.** Selected ordinary-hit wait312→0 cycles proves local intervention. Full T4096 SPM4751.4864→4741.3700 us (0.213% faster), cache unchanged4703.5076. T512 regresses0.75%.

**Takeaway:** this identified local HOL delay was not a large recoverable full-FA cost in the tested configuration. [Report](../../results/cacheflex_flash_attention/demand_isolation_20260922/experiment/REPORT.md).

### FA-20 — MICRO timing alignment and 100-case sensitivity rerun

**N/D, Sep22–23.** Dependent probes match Classic approximately: L1~5 cycles, privateL2~18, LLC~91, SPM~11 including loop overhead. Calibrated fixed T4096 cache/SPM5716.1876/5908.1544 us versus old timing4703.5076/4751.4864. Repeated12 intervention bundles across old/calibrated timing: ingress4/16, demand isolation, copy/load/combined acceleration, response floor, retention, prefetch, network256, guard4096. Calibrated network256 gives cache5187.336/SPM5266.4308 us; fast direct loads yield no meaningful runtime gain; demand isolation slightly regresses (~0.083%).

**Takeaway:** timing calibration changes results; old numerical acceptance does not mean equivalent hardware. Generic network improvement still does not create a large SPM advantage. [Calibration](../../results/cacheflex_flash_attention/timing_alignment_20260922/REPORT.md), [all sensitivities](../../results/cacheflex_flash_attention/calibrated_sensitivities_20260922/REPORT.md).

### FA-21 — Nearly ideal repeated staging oracle

**N/D, Sep23.** T4096 normal cache5716.1876/SPM5908.1544 us; oracle SPM5699.7240 (3.528% own reduction;1.002888x cache), oracle+local-floor5705.2648. 99.609375% of copies use ideal path, no extra SPM SRAM; first claims remain real.

**Takeaway:** even very optimistic elimination of repeated-copy transport did not expose a large latent gain for this V-only mapping. Oracle leaves CPU/dependencies and some first-use behavior intact; not a universal upper bound or implementable hardware. [Report](../../results/cacheflex_flash_attention/ideal_staging_20260923/experiment_v3/REPORT.md).

### FA-22 — SPMLD consumer chain and instruction-fetch response edge

**D/N, Sep23.** Faster load response9→5 cycles leaves median consecutive load issue interval7 cycles. Traces tie it to a19-instruction/76B PV loop and64B fetch-buffer refill cadence. Matched IFETCH response-edge intervention reduces cadence7→5; T4096 cache5716.1876→5660.9784, SPM5908.1544→5715.4256 us. Faster SPMLD after that still gives5718.6604, slightly worse.

**Takeaway:** a concrete throughput mask was identified and relieved, yet another limit remained. Avoid claiming all FMAs wait on coherent operands; observed first consumers often wait on SPM, without affecting stream rate. [Trace](../../results/cacheflex_flash_attention/spm_consumer_trace_20260923/experiment_v2/frontend_followup/REPORT.md), [matched experiment](../../results/cacheflex_flash_attention/ifetch_edge_20260923/experiment/REPORT.md).

### FA-23 — Coherent K/O load lifetime and L1D miss credits

**D/N, Sep23.** Selected K loads spend~51% of lifetime in admission/retry; O-rescale loads~88% after admission. Rejections identify actual L1D miss8 pool, not global sequencer. Credits8/16/32/64 produce cache5660.9784/5142.7492/5373.9252/5009.5188 us and SPM5715.4256/5144.9376/5124.3728/5124.3216. Fast-SPMLD incremental benefit stays ≤0.236% at16/32/64.

**Takeaway:** coherent miss concurrency affects runtime, but increasing it does not reliably reveal a hidden direct-SPM latency gain. Counter retries are not cycles. [Trace](../../results/cacheflex_flash_attention/coherent_load_path_20260923/experiment_v4/middle_REPORT.md), [sweep](../../results/cacheflex_flash_attention/l1d_miss_credit_20260923/REPORT.md).

### FA-24 — Fixed execution/on-chip-memory 2×2

**N/D, Sep23–24.** T4096 cache/SPM us: control5660.9784/5715.4256; execution5645.3448/5542.6592; memory2051.2492/2171.1556; combined1713.3788/1750.0616. Memory bundle enlarges ports/credits/NoC and reduces positive service floors; capacity unchanged.

**Takeaway:** major common on-chip memory headroom exists, much larger than execution-only gain. This is a many-resource diagnostic bundle, not a realizable free optimization, exact bottleneck allocation or SPM-specific win. [Report](../../results/cacheflex_flash_attention/headroom_2x2_v2_20260923/REPORT.md).

### FA-25 — Output accumulator resident in two SPM ways

**N, Sep24; T4096, O64×64 FP32=16 KiB/core, scaledL2.** Best cache5660.9784 us; V-SPM5715.0284; output-SPM6579.4184; identical-output-driver cache6661.4028. Only2048 bootstrap copies; output-SPM1.01246x its matched driver but loses to best cache. SPM issue→wakeup31.94 cycles versus V-SPM9.01, despite both7-cycle accepted service; 10.739M reschedules.

**Takeaway:** accumulator residency is already implemented/tested, not new. Transfer format/driver and dependency costs matter; do not quote only the weaker matched-control speedup. [Accepted V10](../../results/cacheflex_flash_attention/output_resident_v10_20260924/REPORT.md), [control audit](../../results/cacheflex_flash_attention/output_resident_20260924/control_audit/REPORT.md).

### FA-26 — Finite SPM store-to-load forwarding

**N/D, Sep24.** Directed test exercises259 correct forwards. Full FA T4096 OFF=ON6579.4184 us, zero forwards; T512 also exact no-op. Trace shows unrelated-slot loads wait for older stores' unknown addresses, not same-slot store completion.

**Takeaway:** forwarding did not address the observed FA dependency. Zero exercised forwards is a meaningful negative result, not a reason to change acceptance. [Experiment](../../results/cacheflex_flash_attention/spm_forwarding_20260924/experiment_v3/REPORT.md), [trace](../../results/cacheflex_flash_attention/spm_forwarding_20260924/analysis/REPORT.md).

### FA-27 — Early SPM store-address resolution

**N/D, Sep24–25.** Address-only one-cycle unit, metadata in finite SQ, no new data SRAM. T512172.2084→171.8204 us (0.225%); T40966579.4184→6553.7524 (0.390%). T512 reschedules fall~71.2%, without corresponding runtime reduction. Masked-store regressions required a real predicate-write correctness repair first.

**Takeaway:** reduced retry counts are not equivalent to major speedup. This specific disambiguation mechanism was exercised and helped only marginally. [Results](../../results/cacheflex_flash_attention/spm_early_address_20260924/experiment_v4/REPORT.md).

### FA-28 — Persistent V with KV-token partition and explicit reduction

**N, Sep25–26; single-head,512KiBL2, old finite-resource profile.** Thirty-two accepted cases including1/4/8-core gates. T16K query-cache G8 28.5384512 ms versus persistent-V KV-SPM22.4731980 (1.269888x). Earlier~1.47x was superseded when cache grouping improved. Later separate cache tuning reaches19.7961084 ms, undercutting this persistent-V candidate.

**Takeaway:** persistent local ownership can help its matched mapping, but baseline tuning eroded the apparent best-to-best gain. Reduction, replicated query work and different arithmetic schedule matter. [KV suite](../../results/cacheflex_flash_attention/kv_partition_20260925/experiment_v3/REPORT.md), [cache tuning status/accepted cases](../../results/cacheflex_flash_attention/cache_tuning_20260925/experiment_export_safe_20260926/status.json).

### FA-29 — Finite-model analytical tuning and winner PC audit

**N/D, Sep26–27; current finite profile.** Eight accepted pilots +32 gated/search cases, equal budgets across query-cache/query-SPM/KV-cache/KV-SPM. Best measured T4096 cache/SPM1.1892748/1.1673556 ms (1.018777x); T16K19.0266784/18.2140028 ms (1.044618x), both query Br16/Bc512/R768 at T16K. SPM R1024 takes18.8769408 despite33% fewer copies; persistent-V pilot22.3162312 despite24× fewer copies. T16K SPM reduces packets12.1%, L1D misses41.8%, DRAM roughly unchanged. Two coherent K-load PCs account39.56M unexecuted-head samples; V SPMLD only1,154 samples, mean service7 cycles, RAW blocks0.

**Takeaway:** current V path works, remaining K/O/packing work matters; samples are not removable-runtime fractions. Earlier K+V/output-resident negatives MUST accompany proposals to transfer them. Bounded analytical search is not global optimality; initial historical held-out model error17.08% also rules out replacing simulations with exact predicted latency. [Analysis](../../results/cacheflex_flash_attention/finite_model_20260926/winner_mechanism_audit_20260927/REPORT.md), [search status](../../results/cacheflex_flash_attention/finite_model_20260926/analytical_experiment_v5/experiment/status.json).

### FA-32 — Bounded posted near-SPM output accumulation

**D/N T1024 accepted, P larger FA, Sep27.** New opt-in engine performs ordered
FP32 `O=round(round(alpha*O)+contribution)` in existing SPM slots. Four272B source
records, four272B sink records, one256B active intermediate, 16B/cycle local link,
16 FP32 lanes; no full accumulator backing array. Conservative backend uses
existing timed SPM controller reads/writes, not a functional SRAM bypass; bank-
level arbitration/physical area remain uncalibrated. Runtime claims include all
SPM claim, normalization, drain and release costs.

Default-disabled old-binary cache/SPM controls reproduce121.1408/116.3336us and
**all old finite numeric nonhost statistics and output bits exactly**. Both
completion policies and a one-entry/slow-link stress pass518 updates each,
including per-slot recurrence, immediate RAW, overwrite, DSB, WB, release/reuse;
source-full backpressure is exercised. This is not exhaustive protocol proof.

All seven T1024 full-output gates passed, including8192 updates in each hardware
mode. Async154.1748us, sync153.9608us, software-SPM-output152.8888us versus best
current cache120.6400us and frozen standard-SPM116.3336us. This small-case result
is **negative**, not a speedup: posting does not recover output-residency costs
here. Exact output bits match the independent-checked original. T4096 is active;
T16K remains queued. No larger-size conclusion yet.

Full T4096/T16K comparisons are queued for original,
software-fused coherent O, software-fused SPM O, synchronous offload and async
offload. Unlike FA-25–27 this removes core-side accumulator arithmetic and permits
posted completion. Software-fused controls prevent crediting driver fusion to
hardware. [Design](NEAR_SPM_REDUCTION.md),
[live accepted-only report](../../results/cacheflex_flash_attention/near_spm_reduction_20260927/experiment/REPORT.md).

### FA-30 — llama.cpp production reference and Q64 cache packing

**N accepted subset, Sep25–27.** Bit-preserving Q64 packing passes paired gates; cache T1024 example118.5044→117.2236 us. Completed cache search27 cases, best T16K19.7961084 ms on its old profile. Pinned llama.cpp upstream build has no SVE branch in its GEMM helper; separate ported helper uses FP32 SVE. Ported T16K155.6491036 ms; unmodified T4096550.6382972 ms. Upstream T16K was still pending at this ledger's inspection.

**Takeaway:** library identity or compiling with SVE is not proof of optimized SVE execution. ggml FP32 GEMMs/state differ from custom FP16 GEMMs, so do not market these ratios as matched arithmetic. [Cache suite](../../results/cacheflex_flash_attention/cache_tuning_20260925/experiment_export_safe_20260926/status.json), [llama suite](../../results/cacheflex_flash_attention/llama_cpu_20260925/experiment_export_safe_20260926/status.json).

### FA-31 — Capacity-preserving per-way SPM set spreading

**Completed T16K update:** all three repaired-model arms accepted: cache
19026.6784us, original-placement SPM18214.0028us, spread-SPM18120.7800us.
Spreading reduces SPM runtime0.512% (1.005145x); spread-SPM/cache speedup1.049992x.
All1048576 output elements pass independently, with exact common output hashes.
This is a small placement improvement, not evidence for arithmetic offload.

**Sep27 target update:** repaired T4096 matched Br16/Bc512/R256 cache/control-SPM/
spread-SPM1204.8712/1167.3556/1154.85us, all full arithmetic checks pass. Placement
reduces SPM runtime1.07%; spread/cache ratio1.04331x. Cache here is the matched
pilot, not the independently tuned minimum. Three T16K winner-transfer arms
now run concurrently after explicit user authorization; scheduling-only handoff
preserves original two running simulator processes and prevents duplicate spread
launch. Same experiment_retry_v2 status/results; coordinator4127434. No T16K
placement result yet.

**Sep27 repair update (all14 gates accepted; target comparison pending):** isolated
build_retry_v3 adds opt-in stale-owner response NACK and reissue from the existing
requester TBE, retaining the original miss credit. The exact previously failing
Br64/Bc1024/R256/stride512 case now completes at216.726us with all65536 independent
output checks passing, identical output bits to its original unspread reference,
and51NACKs/51reissues. Default-disabled controls exactly reproduce old finite
numeric nonhost stats/output bits; three bridge and two wide-access regressions
pass. Repaired small cache/SPM controls121.1408/116.3336us also pass. Both tiny-resource
gates and exact observer-on/off numeric nonhost stats/output equivalence now PASS.
Matched repaired T4096cache/SPM controls are running; spreading thenT16K queued.
No placement speedup claim or exhaustive liveness proof.
No new data SRAM/request credits/queue slots/VCs. See
[repair design](SPM_SILENT_RETRY_REPAIR.md) and
[accepted-only report](../../results/cacheflex_flash_attention/spm_set_spread_20260927/experiment_retry_v2/REPORT.md).

**D/N prerequisites, X/P target comparison, Sep27.** Current finite model plus
default-off set rotation; stride512 spreads the Bc512 panel's same64KiB over all
1024 sets rather than two ways/half the sets. Default-off T1024 cache/SPM
121.1408/116.3336us exactly match all prior finite numeric nonhost stats and output
bits. Three old bridge regressions and a new VL16 four-way read/store/write-back/
release test pass, including shifted ways and unfenced coherent RAW.

Full-set T1024 Br64/Bc1024/R256 remapping gate deadlocks after ROI begins. Bounded
trace shows invalid-owner silent-forward bounce blocked by its full normal-request
FIFO, LLC forwarding FIFO also full, and stationary request network. Exact
wait-for-cycle reconstruction/repair remains pending. T4096/T16K comparison did
not run; no placement speedup claim. Related FA-03/06/29, MOD-03/04: this changes
per-way distribution, unlike old whole-region offsets, and adds no data SRAM.
Hardware bank-row selection/latency not synthesized. [Plan](SPM_SET_SPREAD_PLAN.md),
[gates/failure evidence](../../results/cacheflex_flash_attention/spm_set_spread_20260927/diagnosis/REPORT.md).

## GEMM and historical FA transport studies

### GEMM-01 — W7 single-core CPU/cache/driver matching

**H/D.** FP16 VL4 paper-capacity/core Ruby approximately0.993x; original-style fused drivers0.985x. VL16 original paper-shaped W7 Classic zero-initialized cache/SPM672.4/569.5 us (1.181x), Ruby584.1/534.6 (1.093x). Zero-initialization was an FMOV compatibility workaround, not rich numerical verification.

**Takeaway:** VL-specific microkernel, core FUs, prefetch/timing and driver matter. Geometry alone is not paper reproduction. Sources: `results/cacheflex_spm_gemm/paper_original_classic_vl2048_*`, `paperdriver_vl2048_*`, `mesi3_*paper*`; individual cases in [index](runs.csv), historical context “Current source state”.

### GEMM-02 — W7 multicore mapping, shared packing and pipelining

**N/H by case.** Initial matched2M×4N cache/SPM243.446/227.133 us (1.0718x; zero-output checks). Later independent mapping search1×8/2×4/4×2/8×1, NEON A packing and pipelined kernels selects4×2 for both:199.990/182.524 us (1.096x). Nonzero pattern-reference counterparts check all802,816 outputs with zero error; use those saved correctness cases for arithmetic evidence. Distinct later rebuilt controls must not be mixed into this pair.

**Takeaway:** shared packing and per-core work decomposition change the baseline substantially. Full output checks, not checksums alone, establish correctness. [Benchmark documentation](../../benchmarks/cacheflex_spm_gemm/README.md), `vl2048_{cache,spm}_best_correctness_ruby_8c_w7` in [run index](runs.csv).

### GEMM-03 — ACL full stack versus ACL scheduler with matched kernels

**D/N by case.** Untouched ACL NEGEMM/packing/scheduler W7 fast-math SVE1 path293.667 us; arithmetic/dispatch caveats differ from later mixed-precision work. Same custom kernels/driver through actual ACL scheduler: cache365.266/SPM335.344 us (1.0892x), full output checks. Faster custom cache remains a contender.

**Takeaway:** scheduler-only matched experiment and untouched full library are different baselines. Do not attribute all full-stack differences to hardware. Sources: `acl_fullstack_fp16_vl2048_ruby_8c_w7*`, `vl2048_{cache,spm}_acl_scheduler_ruby_8c_w7` in [index](runs.csv).

### GEMM-04 — Nonperturbing profiling and instruction pipeline

**D/N by case.** Guest timer markers perturbed execution; symbol-only side ELF/production-PC observation restores exact control/profile equality. SPM226.8736 versus cache248.2904 us; two-way SPM pipeline222.8864 (~1.018x over old SPM). PC categories are additive bookkeeping over marked core cycles, not causal latency allocation. Earlier whole-driver timer percentages must not be mixed with uninstrumented timing.

**Takeaway:** profile exact code and check observer equivalence. A/B movement savings can be masked by common work; packing remains relevant. [Workflow](../../docs/spm_cycle_debug.md), `vl2048_*pc_profile*` cases in [index](runs.csv).

### GEMM-05 — Multicast, bootstrap/claim and staging window

**N for corrected W7 trio; other iterations D/X.** Same rebuilt model: cache204.6328 us, nonmulticastSPM164.3700, multicast161.1668. Multicast cuts copies32,768→14,336, but only1.020x over nonmulticast (1.270x cache). Window256 beat16/64;2M×4N slower than4M×2N in that search. Later claim/bootstrap variants and protocol regressions are individually indexed, not automatically faster accepted winners.

**Takeaway:** reducing duplicate coherent fetches is real, but local reuse/overlap and bootstrap costs limit runtime impact. Preserve nonmulticast control on the same build. Sources: `vl2048_spm_multicast_*`, `*_best_correctness_multicastgem5_8c_w7` in [index](runs.csv).

### LEG-01 — Historical scalar versus multicast FA and query interleaving

**H/D, early VL4 FA; shared conversion defect discovered later.** Receiver-ready hybrid T128/T25661.796/152.539 us; interleaved query blocks/two softmax rows58.506/141.235. Matched old cache68.982/165.751. Later scalar-only sweeps also preserved.

**Takeaway:** evidence that scheduling/overlap changes this instruction stream; **not numerically validated FA speedups**. Do not quote1.179/1.174x as current correct FA. Sources: `spm_konly_multicast_*`, `interleave_*`, [scalar comparison](../../results/cacheflex_flash_attention/scalar_matched_20260907/README.md), historical context “Pending work”.

### LEG-02 — Historical SPM ways and K+V pinning

**H/D.** K-only eligible ways3/4/6/8 all146.582 us when actual claimed slots unchanged. Six-way K+V pinning177.498 us, with private-L2 misses+22.2%, packets+41.6%, despite slightly fewer L1D misses.

**Takeaway:** eligibility is not occupancy; unused ways do not consume capacity under lazy claims. This old arithmetic test already warned against indiscriminate K+V pinning. Sources: `waysweep_konly_*`, `waysweep_kvpin_*` in [index](runs.csv).

### LEG-03 — Batch SPMCP and decoupled 2D DMA

**H/D.** Three-line true-LSQ batch T128/T25663.070/158.664 us versus scalar hybrid61.796/152.539 (2.06/4.02% slower). DMA pinned K+V173.410 us atT256 versus scalar177.498, still worse than K-only146.582. Independent source-line translation and blocked-first-issue fixes required for DMA correctness.

**Takeaway:** command reduction does not necessarily reduce transport or critical-path delay; grouping may reduce overlap. Batch/DMA were implemented already, but old FA hashes do not establish attention arithmetic. Sources: `*batch3_lsq*`, `spm_kv_dma_pinall_final*`, historical context “Pending work”.

### LEG-04 — Broad DMA/core/tiling/ports/compute sweep

**H/D accepted subset; invalid Bc128/mapping failures X.** Swept DMA1/2/4/8 ports, Br32/64/128, Bc64/96/128, ROB128…1024 and associated queues, large caches and SIMD latency. Fixed accidental claims of unused V. Paper-core T256 scalar/DMA141.761/140.610 us; four injection ports138.154. Enlarged-window best132.979 versus matched enlarged cache145.787. More ports/window did not monotonically help. Separate scalar core probes: T12861.796→58.832 large window,58.522 one-cycle SIMD; T256152.539→148.082/145.841.

**Takeaway:** these are historical sensitivity clues, not validated current FA or proof of fundamental impossibility. Shared hardware upgrades need matched cache controls. Sources: `dma_*`, `noclaimv_*`, `ablate_*`, `scalar_*interleave*`, historical context “Pending work”.

### LEG-05 — Early FA VL4/VL16, tiling, causal factorials and residency

**H/D, Sep7–11 before arithmetic repair.** Includes clean-comparison, native-VL16, tile24-case, remaining/deeper searches (including K+V), factorial capacity/copy/load/window/SIMD tests, resident QK and eviction probes. Historical matched hashes/counters allow instruction-stream/model diagnostics, but dense conversion defect invalidates claims of correct attention performance. Later corrected FA-01–07 supersede workload conclusions.

**Takeaway:** keep source evidence; do not revive impressive old ratios or treat all failed/stalled cases as slow valid runs. Sources: [attribution](../../results/cacheflex_flash_attention/attribution_20260907/REPORT.md), [clean VL16](../../results/cacheflex_flash_attention/clean_vl16_20260908/REPORT.md), [causal factorial](../../results/cacheflex_flash_attention/causal_20260910_v3/REPORT.md), [residual](../../results/cacheflex_flash_attention/residual_retry_20260911/REPORT.md), [deeper sweep](../../results/cacheflex_flash_attention/vl16_deeper_20260908/queue_v2/summary.json).

## AI workload search and SwiGLU

### AI-01 — Initial broad proxies / promising-shape screens

**H/D by case.** Grouped GEMM, expert/MoE-like and quantized projections, GQA/PV, FFN, batched transformer, beam search, speculative decode, LoRA-base, diffusion projection and pointwise convolution cases are preserved. Many are only a GEMM proxy, single-core despite broad labels, or use checksum-only/old precision/timing checks. The eight-shape `promising` GEMM screen is not the paper's W1–W7 suite. `real_operators/diffusion_ffn_*` implements one projection+GELU+residual, not full SwiGLU/transformer FFN.

**Takeaway:** shape-level hints motivated the later mapping-aware screen; no full-model or routed-MoE claim. Use per-case printed result lines/epochs in the index, not a pooled headline geomean. [Original MLP audit](../../benchmarks/ai_workloads/MLP_AUDIT_20260909.md), `results/ai_workloads/{promising,real_operators,moe_*,quant_*,gqa_*,pv_*,*baseline*,*spm*}` in [index](runs.csv).

### AI-02 — Clean full FP16 SwiGLU and ACL audit on old 32 MiB LLC

**N, Sep9.** Full custom MLP T64/D512/F1536 cache268.2460/SPM155.7484 us (1.7223x); T128312.9880/199.9308 (1.5655x), selected2M×4N. First-call weight packing included. Actual LLC32 MiB, not4 MiB. Pristine ACL first/warm T64 1270.5316/350.5656 us; T1281553.1040/548.7324. ACL SVE dispatch verified; activation/KC rounding schedules differ but independently audited.

**Takeaway:** valid first-call operator results, not warm best-to-best or standard4MiB configuration. Avoid comparing warmed ACL against first-call SPM or claiming all gaps architectural. [Custom](../../results/ai_workloads/mlp_clean_20260909/REPORT.md), [ACL](../../results/ai_workloads/mlp_acl_20260909/REPORT.md).

### AI-03 — Reusable weights and ordered MLP tuning / fusion

**N accepted subsets, Sep9–10.** Persistent cache packing reverses earlier benefit: T64 warm cache128.1840/SPM143.7568 us (0.89167x); T128176.8480/186.7868 (0.94679x). Longer T128175.7172/185.0764. Independent KC/MP/8-/16-/24-row/fusion stages have saved completed summaries; original queues hit prefetch-transient and duplicate-wakeup bugs. Corrected repeated-call regression median172.6676/181.7432 us. Do not pool builds.

**Takeaway:** first-use packing can dominate apparent gains; valid completed subsets survive queue failures, but missing later stages are not evaluated. [Ordered study](../../results/ai_workloads/mlp_study_20260909/REPORT.md), [repair/regression](../../results/ai_workloads/mlp_lsq_fixed_exact_20260910/REPORT.md), individual stage summaries via [inventory](INVENTORY.md).

### AI-04 — Mapping-aware broad screen on 4 MiB LLC

**N, Sep17–20; bounded synthetic FP16 operator screen.** Eight shapes,80 configurations/160 timing cases plus gates; per-core grouping/team/M-N mapping, KC128/256, separate/combined SwiGLU. Independent first/warm winners:

| Workload shape (M or T, K/D, N/F) | First-call cache/SPM | Warm cache/SPM |
|---|---:|---:|
| Grouped distinct weights32,512,512 | 5.201x | 1.257x |
| Grouped distinct weights64,1024,1024 | 4.198x | 1.516x |
| Shared-weight batch32,512,512 | 1.278x | 0.722x |
| Shared-weight batch64,1024,1024 | 1.221x | 1.064x |
| Pointwise1×1,96,512,512 | 2.160x | 0.524x |
| Pointwise1×1,784,1024,1024 | 1.361x | 1.083x |
| SwiGLU64,512,1536 | 3.255x | 1.261x |
| SwiGLU128,512,1536 | 2.767x | 1.254x |

**Takeaway:** grouped distinct-weight projection and SwiGLU merit follow-up; small shared/pointwise warm operators can lose. First-call gains are much larger than warm. Not routed MoE/full models, final convergence or paper-ready winners. [All accepted cases/winners](../../results/ai_workloads/mapping_screen_export_safe_v2_20260917/summary.json).

### AI-05 — Small SwiGLU warm confirmation and partially completed tuning

**N completed cases / X incomplete planned search, Sep20–21.** T64/T128 numerical confirmations pass; warm spread exceeded2% in a case. User removed the variability gate; spread still reported. Resumed queue completed31 tuning cases and2 ACL audits before user redirected to model shapes. ACL T64 warm665.7208 us on the4MiB profile. Planned128-case sweep/finalists were not all completed.

**Takeaway:** variation is not automatically a correctness failure; a warm sequence is not independent repeated simulations. T64 tuning must not be sold as representative model-size tuning. [Original status](../../results/ai_workloads/swiglu_confirmation_20260920/status.json), [resumed accepted subset](../../results/ai_workloads/swiglu_confirmation_resumed_20260921/status.json), [stop marker](../../results/ai_workloads/swiglu_confirmation_resumed_20260921/STOPPED_BY_USER.json).

### AI-06 — Model-shaped FP32-accumulate SwiGLU pilots

**N accepted targets, Sep21–26; eight cores,T128/512/2048,D2048,F8192.** Warm-last-three medians cache/SPM ms: T12814.5644628/12.7298928 (~1.144x); T51260.9547584/57.7280660 (~1.056x); T2048261.9656328/231.6088388 (~1.131x). First T2048269.0210764/231.7162728. Full export/reference/hash checks accepted for both custom paths. ACL warm T12836.1960136 ms, T512141.0205196; T2048 ACL did not complete (disk guard). ACL mixed-precision dispatcher selects NEON on this SVE1 system, not SVE2 mixed kernel.

**Takeaway:** meaningful recurring benefit survives prepacked cache weights at model dimensions, but this is untuned synthetic-input operator evidence on an older model. Faster custom SVE cache, not slower NEON ACL alone, is the comparator. [Accepted subset and failure](../../results/ai_workloads/swiglu_model_pilot_v4_20260921/status.json), [precision audit](../../results/ai_workloads/swiglu_model_pilot_v2_20260921/ACL_PRECISION_AUDIT.md).

### AI-07 — SwiGLU one-core comparison

**N, Sep25–26; T2048,D2048,F8192, same older pilot platform.** First cache/SPM1252.7414644/1204.5756984 ms (1.040x); second1215.6929784/1204.4570296 (~1.0093x). Eight-core warm benefit~1.131x is larger, not smaller, in these tested mappings.

**Takeaway:** multicore does not universally destroy CacheFlex benefit. One-/eight-core mapping and number of warm invocations differ; two calls do not establish converged warm scaling. [Accepted cases](../../results/ai_workloads/swiglu_single_core_v2_20260925/status.json).

### AI-08 — SwiGLU/FA reuse comparison and accumulator oracle

**D plus N small gates / P missing target oracle.** SwiGLU32KiB weight panel reused across1024 rows versus old FA16KiB V panel64 rows; source-derived reuse16×. Both copy lifetimes~360cycles. SwiGLU logical coherent accumulator traffic~6000MiB/call atT2048; not measured DRAM attribution. Fourth-call SPM accumulator-load PCs account71.5% of unexecuted-head samples, not runtime. Selective early-read-value oracle passes small gates with late-value checks; full target intervention stopped at disk guard, so **no completed causal gain measurement**.

**Takeaway:** accumulator blocking is a justified target, but has not been proven by the unfinished oracle; benefits to cache must be tested too. [Reuse audit](../../results/ai_workloads/swiglu_fa_analysis_20260925/REPORT.md), [PC profile](../../results/ai_workloads/swiglu_accumulator_oracle_20260925/pc_profile.json), [oracle queue](../../results/ai_workloads/swiglu_accumulator_oracle_20260925/experiment_v4/status.json).

### AI-09 — SwiGLU current-finite-model transfer

**Sep27 repaired-model continuation:** new swiglu_retry_transfer_20260927 repeats
all4small gates successfully on build_retry_v3 with retryON, stride0. T512 cache
andSPM active in parallel with FA, same AI-06 guest/input/tiling/five-call policy.
No new model-size timing yet. This revalidates earlier results under the bounded
model and progress repair; it is not a tuning or extra-storage experiment.
Old stopped output directory remains intact. T2048 not queued in this restart.

**N small gates / X/P targets, Sep27.** Frozen AI-06 cache/SPM guests, inputs and
five-call policy transferred to MOD-03 hardware. Four T24/D256/F512 numerical,
configuration, full-chain accuracy, export-bit/hash, copy and occupancy gates
pass (KC128 separate and KC256 combined). This is model revalidation, not tuning.
T512 pair was stopped shortly after launch to free slots for FA-31 deadlock
diagnosis; T2048 not launched. No new model-sized speedup. Original model and
accepted older timings unchanged. [Queue evidence](../../results/ai_workloads/swiglu_finite_transfer_20260927/status.json).

## Model fidelity, correctness and MICRO reproduction

### MOD-01 — Finite hardware-resource guards and return-lane ordering

**N/D accepted repaired version, Sep21.** Hardware-bounds v5 passes defaults, stress and guard1024/4096 equivalence; T4096 cache4653.9208/SPM4965.8776 us. Earlier finite queues deadlocked because recall returns shared request FIFOs or PUTX was overtaken by INV_ACK/GETX. Repairs separate return lane using fixed total local budget and order returns before requests.

**Takeaway:** finite-buffer deadlocks are model/protocol bugs, not workload performance results; unlimited queues can hide them. Distinct guard equality establishes nonbinding bookkeeping only for tested cases. [Accepted v5](../../results/cacheflex_flash_attention/hardware_bounds_20260921/experiment_v5/resource_audit/REPORT.md).

### MOD-02 — MICRO miss/writeback pool translation and geometry

**N/D, Sep21–22.** MICRO-derived8/20/32 miss and8/8/16 writeback pools, bridge WB accounting, SPMWB source/readiness/destination RAW repairs. All v6 gates/guard repeats accepted; fixed T4096 cache4703.5076/SPM4751.4864 us. Geometry512KiB L2/no-retention pair2023.3620/2068.8376 us (~2.25% SPM slower). Both have1024/4096 exact numeric nonhost equivalence.

**Takeaway:** hardware limits should be explicit, not inferred from an arbitrary sequencer budget. Geometry and policy bundle changes performance substantially without reproducing paper gains. [Measurements](../../results/cacheflex_flash_attention/micro_resources_20260921/MEASUREMENTS.md).

### MOD-03 — Current bounded model and finite-network deadlock repair

**N/D, Sep26–27.** Finite transport stress exposed request-vnet dependency cycle. Partitioned existing four request VCs into2 private→LLC/2 directory-bound without adding bandwidth/storage. Corrected experiment_v5 all31 cases accepted, including bridge regressions, stress, nominal/guard equality and target pilots. Analytical queue completed32 cases afterward.

**Takeaway:** current usable profile is build_v3/experiment_v5, not earlier failed stress builds or unbounded analytical seeds. This is a modeling correctness assumption, not a CacheFlex speedup mechanism or exhaustive deadlock proof. [Diagnosis](../../results/cacheflex_flash_attention/finite_model_20260926/diagnosis/REPORT.md), [validation](../../results/cacheflex_flash_attention/finite_model_20260926/experiment_v5/status.json), FA-29.

### MOD-04 — Model fidelity audit / bank arbitration limitation

**D, Sep22.** Global guard/pure-SPM cache-blocking not active in audited run; controller transition budgets rarely full. Old response timing differs from Classic; calibrated later in FA-20. `resourceStalls` disabled and protocol lacks SRAM reservation hooks: explicit SRAM bank contention is not modeled. Private-L2 WB-credit lifetime also differs from Classic.

**Takeaway:** do not invent a measured bank-arbitration fraction or call nominal timing match complete hardware calibration. [Audit](../../results/cacheflex_flash_attention/model_fidelity_audit_20260922/REPORT.md).

### MOD-05 — Arithmetic conversion, toolchain and functional export audits

**D/N regression evidence, Sep11–26.** Dense FP16 SVE lane conversion wrong in old FA; ZIP/UZP repair independently checked. GCC15 lane probe reproduces original errors240/4032 atVL4/16, repaired0: not merely GCC11. Separate FMOV encoding incompatibility is toolchain-related. Several post-ROI exports lose filename/data visibility; bit-preserving export adapters reproduce exact ROI counters and restore independent output checks. Failed exports are not automatically arithmetic failures, nor may they be accepted without repair.

**Takeaway:** hashes alone insufficient; check actual output against independent precision-aware reference. Preserve failed evidence. [GCC15 conversion](../../results/cacheflex_flash_attention/micro_conversion_gcc15_20260921/REPORT.md), [toolchain](../../results/cacheflex_flash_attention/micro_toolchain_audit_20260918/REPORT.md), [latest export repair](../../results/cacheflex_flash_attention/export_repair_20260926/README.md).

### MOD-06 — Classic versus Ruby corrected MICRO FA

**N for accepted common-core comparison; not exact paper reproduction.** Corrected GCC15 VL16 T4096/Hq32/Hkv8/Br64/Bc384 Ruby454.8482616/389.5600352 ms (1.167595x); Classic ratio1.168032x, full8,388,608 outputs and cross-model bit equality. Common core FU counts differ from upstream sweep. Later paper-FU original-driver Ruby465.4200592/398.7554152 ms (1.167x); common pthread one-core400.4154272/379.4197980 (1.055x). Four-core head scaling fails functional read, eight-core/larger-T not reached in that chain.

**Takeaway:** wrapper and CPU settings affect ratio. Latest repository is not a provenance record for the paper's1.34x winner. Upstream VL16 stages K+V but PV reads coherent V; label actual consumers accurately. [Audit](../../results/cacheflex_flash_attention/micro_paper_audit_20260921/REPORT.md), [head scaling](../../results/cacheflex_flash_attention/micro_head_scaling_20260921/REPORT.md), [revalidation status](../../results/cacheflex_flash_attention/micro_gcc15_export_revalidation_v2_20260919/status.json).

### MOD-07 — Untouched MICRO timing-only and repository arithmetic audit

**H/D/X.** Nineteen FA source files (17unique) audited across master/rebuttal; no independently arithmetic-verified path to paper result found. Untouched GCC15 timing-only queue has some completed cases/smokes, explicitly `arithmetic_validation=false`, then user cancelled. Corrected paper-core reconstruction also has accepted small gates but incomplete broad target sweep.

**Takeaway:** neither a checksum nor an incomplete timing-only screen reproduces the paper claim. Do not automatically resume cancelled MICRO work. [Source audit](../../results/cacheflex_flash_attention/micro_fa_repository_audit_20260921/REPORT.md), [timing-only report](../../results/cacheflex_flash_attention/micro_untouched_timing_v2_20260921/REPORT.md), [corrected reconstruction](../../results/cacheflex_flash_attention/micro_paper_reproduction_v2_20260921/REPORT.md).

### MOD-08 — Other correctness repairs and instrumentation regressions

**D; useful even when a performance sweep stopped.** Includes SPM canonical-address/IFETCH alias repair; dynamic claim/release and GETX snapshot ordering; missing prefetch transient stalls; last-dependency SPMLD wakeup; software-prefetch credit-release retry wakeup; SPMWB metadata/readiness/destination RAW; vector SPMST payload/way offsets/coalesced side effects; predicate masks; precise bridge draining; timestamp observer side effects. Repaired directed tests and exact default controls are evidence; failed build/timeout/crash timings are not.

**Takeaway:** distinguish implementation bugs from architectural tradeoffs. Do not suppress failed reads, relax numerical tolerances, or enlarge deadlock timeout to obtain a result. Evidence spans [GETX regression](../../results/cacheflex_flash_attention/q_getx_fix_20260916_v3/status.json), [MLP LSQ repair](../../results/ai_workloads/mlp_lsq_fixed_exact_20260910/REPORT.md), output-resident/early-address reports above, and the inventory's smoke/regression records.

### L3-01 — Shared physical-L3 SPM prototype (2026-09-28)

**N/D; T4096 shared arm failed, replacement repair pending validation.** Related FA-08 and FA-25–27/32 concern
coherent LLC retention or private-SPM output residency/reduction, not shared
LLC storage. This experiment repurposes existing LLC slots, preserving normal
coherent eviction on claim, with native shared load/store/copy/writeback/release.
No added data SRAM or reduction hardware. Eight-core VL16 SE, 4MiB total LLC,
single-writer/synchronized-reader software contract. Shared vector transfers
pay full NoC payload costs; common optional64B/cycle/bank service model and35cycle
LLC service delay are explicit assumptions, not calibrated hardware.

Eight host checks pass; build_v4/guest_v4, experiment_v7. Default-off controls
exactly reproduce121.1408/116.3336us, prior nonhost stats and output bits. Normal
and two-entry-queue eight-core sharing/dirty-data/reuse tests PASS zero errors
over four rounds. Repaired missing instruction-prefetch stale response and
unclaimed speculative-read semantics; failed evidence preserved. Dependent
latency probes, then full-output FA T1024/T4096 cache/private-SPM/shared-output
pilots. A useful result is correct finite-resource progress with plausible
latency, not an assumed speedup. FA keeps private V and moves CPU-updated O to
shared slots; not best-to-best tuning. T1024 all three accepted with full
independent arithmetic and exact prior output bits: cache120.9356us,
private-SPM117.3456us, shared-L3-output173.4896us. This direct remote-output
residency transfer is slower, not evidence for/against near-L3 reduction.
All seven latency probes pass; doubling shared response delay35->70cycles
increases256-dependent-load ROI9.7652->13.3480us, matching expected sensitivity.
T4096 cache1194.3868us/private1167.5820us accepted; shared-output failed with
TreePLRU index12/size12, not a timing result. GDB identified physical-tree leaf
indices being applied to compacted eligible-way candidates. The L3-02 follow-up
repairs filtered selection and reruns all arms. This can change victim choice
under claimed-SPM pressure, so retain old shared measurements as historical
evidence, not the final repaired comparison. No paper-ready speedup claim.
[Progress](L3_SPM_PROGRESS.md), [design](L3_SPM.md),
[status](../../results/cacheflex_l3_spm/initial_20260928/experiment_v7/status.json).

### L3-02 — Near-shared-L3 FP32 accumulator reduction (2026-09-28)

**N/D; all five T1024 and T4096 arms accepted. Retry-isolation follow-up pending.** Unlike L3-01
remote CPU read/modify/write and FA-32 private-SPM offload, retains O in existing
shared LLC slots and sends scale plus256B contribution to a bank-local engine.
Four source and four in-flight records/core, four sink records/bank, serialized
16-lane engine, separately rounded multiply/add;35cycle read/9cycle arithmetic/
35cycle write-response and common64B/cycle bank port. Real Garnet payload costs,
no new accumulator SRAM. Assumed timing, not calibrated implementation costs.

Build_v1 experiment_v2 disabled controls exactly reproduce L3-01 T1024 output
bits and nonhost counters. Directed async test exposed lost bank wakeup; trace
identified clearing the service timer after checking internal work. Build_v2
repairs that source ordering plus L3-01's filtered PLRU bug. Experiment_v3 passes
all3 disabled exact controls and all3 async/sync/one-entry correctness gates
(1040 commands each, zero errors); native PLRU24tests pass. Five-arm T1024/T4096
cache/private-SPM/L3-storage/L3-async/L3-sync FA screen is running. Full output references, exact
command accounting and finite-resource/config guards required before acceptance.
Do not infer FA benefit from passing directed tests or reduced transfer bytes.
First accepted T1024 FA: new-ELF cache121.1296/private118.6076us,
L3-storage171.0292us, L3-async151.2172us, L3-sync152.6612us. Full independent numerical and exact
output-bit checks pass;8192commands complete and finite occupancies within4.
Async cuts11.58% latency relative to remote storage-only, but does NOT beat
standard private-SPM117.3456us or cache120.9356us original-ELF controls on the
same platform. Network flits710266->561441, still above original private344259.
Takeaway: moving arithmetic to L3 recovers part of remote-output cost; no net
benefit over the strongest accepted normal CacheFlex implementation at this T.
T4096 final: cache1202.3908/private1168.2400/L3-storage1305.9208us;
async and sync both1208.3864us, full checks/32768commands each. Offload recovers
7.469% versus storage-only but remains3.436% slower than matched privateSPM.
User-authorized follow-up isolates reduction-only retry backpressure from
coherent requests with unchanged capacities/timing/bandwidth/guest, unlike
FA-14's copy-path isolation. build_v3/retry_v2 passes default exact and enabled
no-reduction exact controls plus3 isolated directed gates (1040commands each,
zeroerrors). Coherent progress while reduction-blocked159/150/997 confirms the
intended decoupling; FA isolated T1024/T4096 still pending. Retry_v1 stopped on
an incorrect host counter identity (mixed ordinary-load/command ordering-retry
count); sourceFull/isolated-command equality held and numerical checks passed.
No simulator or numerical tolerance changes in the runner accounting repair.
Retry_v2 T1024 isolation accepted: async151.2172->147.3932us (2.529% lower
latency), sync152.6612->147.2528us (3.543%). Same guest/bounds/timing/bandwidth,
full math/hash/config checks.126340/126556 coherent requests progress while
reduction admission is blocked. This fixes a real coupling and saves measurable
runtime, but still does not beat privateSPM117.3456us. T4096 isolation pending.
UPDATE: T4096 isolation accepted: async1173.7940/sync1175.7928us,2.863%/2.698%
lower latency versus1208.3864us. Async still0.475% slower than matched private
1168.2400us. Retry queue complete; full checks pass.
[Progress](L3_REDUCTION_PROGRESS.md),
[status](../../results/cacheflex_l3_spm/reduction_20260928/experiment_v3/status.json).

### L3-03 — Bounded bank pipeline and typed FP16 contributions (2026-09-28)

**Complete; all T1024/T4096 arms accepted.** L3-02 showed retry
isolation recovers a few percent but not a net private-SPM win. Source audit
identified serialized35cy read+9cy arithmetic across independent output rows,
and256B FP32 contributions originating from128B FP16 PV results.
Opt-in bounded pipelining retains four sink entries, shared SRAM port64B/cy,
read/arithmetic latency, row ordering and final completion hazards. Adds
explicit4cy initiation / shared future ALU reservation, not unlimited ALUs.
Typed compact transport retains FP32 accumulators/nonfused rounding;128B
contribution+16B header,6cy conversion stage, same network bandwidth. CPU/source
capture still256B. No new accumulator SRAM. Model assumptions not RTL calibration.
Build_v2/guest_v2,26source guards pass. experiment_v2 starts old-ELF exact gates,
then pipeline-only, typed/mixed directed tests and matched2x2 factorial with
cache/private controls atT1024/T4096. Stop on numerical/config/resource failure.
UPDATE accepted old-ELF hardware-only pipeline comparison: T1024
147.3932->147.1656us (0.154% lower latency); T4096
1173.7940->1175.2736us (0.126% higher). Full math/hash/config checks pass and
maximum bank concurrency3/2 proves overlap. Bank serialization alone is not a
material removable end-to-end cost in these tested configurations. Typed/mixed
directed tests pass; matched FP16/combined FA performance remains pending.
UPDATE all T1024 matched-ELF arms accepted: cache119.6668/private115.5796us,
serializedFP32 147.2108, pipeline147.0968, FP16-only143.8136, combined141.1488.
Full oracle/exacthash/config/bounds pass. Isolated pipeline0.077% saved latency,
FP162.308%, combined4.118%(1.042948x); pipeline after FP161.853% additional
latency reduction. Thus interaction exists, but combined still22.123% slower
than privateSPM. No net FA benefit claimed. Total flits561997->496254; contribution
wire bytes272->144 per command, FP32 accumulation unchanged, CPU widening also
removed. Eight directed tests pass including mixed precision and one-entry stress.
T4096 six-arm factorial COMPLETE: cache1200.6220/private1170.4900us,
FP32base1177.7440/pipeline1180.3720/FP16-only1159.5416/combined1163.4248.
FP16-only saves1.546% vs FP32; adding pipeline makes it0.335% slower.
Best compact variant is only1.00944x vs matched private, not a large net gain.
See experiment_v2/REPORT.md/EFFECTS.md
under results/cacheflex_l3_spm/pipeline_20260928; coordinator1286011.
[Progress](L3_PIPELINE_PROGRESS.md).

### L3-04 — First-update initialization and near-L3 finalization (2026-09-28)

**Complete; all T1024/T4096 arms accepted.**
Related L3-02/03. Remove lifecycle transfers rather than increase queues: first
update claims slots through normal coherence and initializes from explicit +0;
finalization normalizes/converts remotely, writes two64B coherent lines per row,
and releases the four accumulator slots. Same corrected FA mapping/microkernels.
No extra SRAM. Source SPMWB translation/hazards retained, two-slot source hazard.
Finite bank sink shared with reductions, read35cy/16-lane multiply+narrow8cy,
response35cy and same64B/cy bank port. Explicit modeled hardware assumptions.
Important confound: existing coherent SPMWB waits directory/DRAM acknowledgment,
so finalization includes write-through output policy and an extra controller hop.
Separate init/final/both ablations make this visible. Best prior engine fixed
within each T: pipelinedFP16 at1024, serializedFP16 at4096. Exact old-ELF controls,
six directed math/coherence/reuse/one-entry tests, then matched six-arm FA pairs.
Frozen root results/cacheflex_l3_spm/lifecycle_20260928, runner run_l3_lifecycle.py.
build_v5/experiment_v4 passes31source checks and ALL9runtime gates (three exact
controls, six directed init/final/both/one-entry/sync/serial). New finalization
exposed/fixed metadata masking, descriptor-size assertion, and real cross-SQ
SPMWB source RAW ordering bug; bounded trace and prior failures preserved.
Existing SPM SQ/shadows now prevent WB overtaking older overlapping stores,
without a global fence or extra buffer. All6 same-ELF T1024 cases pass full
65536element independent oracle/exact output/config/copy/resource checks:
cache121.0324/private115.8060/reductionbase140.9140/init128.8536/final168.8404/
both156.5028us. Init saves8.559% versus reduction; final worsens19.818%; combined
11.063% slower than reduction,35.142% slower than private. No net new benefit.
Final2048WB add exactly131072B DRAM traffic (257536->388608); slight flit
reduction does not yield faster ROI. WBmean281Rubycycles is not an additive
runtime share. This outcome includes write-through/DRAM-ack policy, so it does
not isolate near-L3 normalization arithmetic. First-update fusion is useful;
current finalization path is not. T4096 was unmeasured at that earlier entry.
Final T4096 accepted results: cache1191.1572/private1160.5792/base1158.4664/
init1133.0080/final1270.6976/both1249.3676us. Initialization saves2.198% versus
reduction and2.376% versus matched private; finalization worsens9.688% versus
reduction and combined is7.650% slower than private. All262144 elements pass
the full independent reference and exact prior-output/config/resource checks.
This is a small net win for initialization atT4096, not a general L3 offload
or T16K performance claim. experiment_v4 is completed_requires_analysis.
[Progress](L3_LIFECYCLE_PROGRESS.md).

### L3-05 — Direct PV epilogues and L3-aware kernel tuning (2026-09-28)

**P; implemented, validation running.** User authorized sequential experiments
after overlap and server-utilization checks. Related FA-06/07/29 already tuned
kernel shapes and grouping, but not against the L3 reduction cost model.
L3-03 removed CPU widening and compacted contributions; L3-04 fused first-update
initialization. Both still wrote PV results to a CPU temporary and reloaded
them before reduction submission. No direct register epilogue was found in
the ledger, inventory, or the inspected reduction builders/drivers.

New software-only guest replaces the PV store/reload epilogue with register
commands; includes comparable fused CPU accumulation for private-SPM/cache.
FP16 GEMM loop unchanged; FP32 multiply/add remains nonfused. Same frozen
lifecycle build_v5, geometry, queues, bank/network timing and output policy.
Then bounded Bc tuning, then Br/query grouping and independent-row submission
order. Earlier tile ideas are revalidated/transferred, not claimed as new.
Full independent numerical checks, exact same-Bc output bits, copy/command/
transport/configuration/resource checks required. Different Bc has its own
precision-aware reference. Current best lifecycle T4096 is1133.0080us versus
matched private1160.5792us; decision requires lower total ROI and improved
comparison against independently tuned private SPM. No new accepted gain yet.

Preflight97.5%CPU idle/712GiB RAM available/no swap; cap2 new sims alongside
3 existing. Launches wait on CPU/memory/disk headroom. Stage failures stop
downstream work. guest_v1 parsing failure never simulated; guest_v2 compiled.
[Plan](L3_KERNEL_TUNING.md),
[status](../../results/cacheflex_l3_spm/kernel_tuning_20260928/experiment_v1/status.json).

**Accepted T1024 subset:** all six buffered/direct cache/private/L3 cases pass
full independent65536element arithmetic, exact prior-output bits, configuration,
copy/command/transport and finite-resource checks. Same-ELF direct epilogue:
cache121.7416->119.7604us (1.627% lower), private117.5920->115.7564 (1.561%),
L3init128.8200->127.0804 (1.350%). This improves both paths but does not create
a small-case L3 win (still9.783% slower than matched direct private).
One-entry full-FA direct-L3 stress also passes8192updates/1024initializations,
zeroerrors;150.2640us is not a target-performance result. All six generated
compute loops match the frozen FP16 assembly exactly. The direct epilogue
emits independent rows in paired order; later scheduling compares sequential
order, so do not attribute the entire delta solely to removed temporary bytes.
T4096 and subsequent Bc/group/order results remain pending.


**L3-05 completion update:** all 25 cases accepted with full checks.
T4096 cache: direct epilogue: 1193.4408->1166.0656us; 2.294% lower latency (1.023477x).
T4096 private: direct epilogue: 1156.6056->1145.2836us; 0.979% lower latency (1.009886x).
T4096 l3: direct epilogue: 1137.6424->1117.5820us; 1.763% lower latency (1.017950x).
T4096 current best L3 vs cache: 1166.0656->1041.9376us; 10.645% lower latency (1.119132x).
T4096 current best L3 vs private: 1078.0056->1041.9376us; 3.346% lower latency (1.034616x).
Bounded software search; original hardware resources unchanged. Read the effects report for negative and positive comparisons.

### L3-06 — Qwen-shaped causal GQA on the current L3 kernel (2026-09-28)

**P; compiled, runtime gates pending.** User requested starting immediately with
a real model configuration. Published Qwen2.5-0.5B-Instruct Hq14/Hkv2/d64,
batch1, causal prefill, eight cores/VL16. Retains the checked FP16 GEMMs/FP32
online state: this is model-shaped attention, not full Qwen BF16 inference.
FA-06/07 already tested older multihead tuning; FA-08/09 found near parity at
large T despite traffic savings. New case transfers L3-05 direct epilogues and
current finite hardware to causal GQA and balanced head/query-group tasks.

No simulator changes. Same frozen lifecycle build_v5, same resource bounds.
One new ELF for cache/private/L3, Br16/R256; inherited cache Bc512 and private/
L3 Bc256, no new tuning sweep. Separate K/V heads physically stored once, seven
query heads per KV head. Core-owned L3 bank slots are drained/released before
each task reuses them. Causal future blocks skipped at group granularity;
future positions in the boundary blocks explicitly masked before softmax.

Sequence: old single-head exact-output compatibility; small 14Q/2KV causal
full-output gates plus depth1; T1024 validation; T2048/T4096 three-arm targets.
Every output must pass independent precision-aware reference; same-Bc exact
bits, dense small-case oracle, first-row identity, counts/config/resource bounds.
Full ROI includes packing/staging/masking/normalization/drain. Decision is lower
complete latency versus cache and separately private-SPM. No accepted new
performance yet. Max2 new simulators with CPU/memory/disk launch guards.
guest_v1 export-padding guard failed before simulation; v2 compiled; v3 moves
diagnostic printing after joined workers and passes unchanged exporter guard.
Three host tests pass for task partition/counts, causal future independence,
KV-head isolation and original input-pattern compatibility.
[Plan](QWEN_GQA_L3.md),
[status](../../results/cacheflex_l3_spm/qwen_gqa_20260928/experiment_v2/status.json).

**Scheduling correction before targets:** experiment_v1 accepted two one-head
exact-output controls, then was deliberately stopped to rotate query-group
order across heads. Plain round-robin tied each core to the same causal-prefix
length; rotation bounds maximum core row-update work to <1.15x the mean at the
targets. Same mapping in every arm, unchanged arithmetic/model. Partial v1
evidence retained. guest_v4 SHA
f48abf7caaf2c5044e1b1dea662017cf02a528c32f730ec2640baedfd3a09879;
experiment_v2 repeats every gate.

**Accepted initial v2 gates:** all three one-head compatibility controls pass
full independent arithmetic/config/count/resource checks and exact prior output
bits. All three T256 Hq14/Hkv2 causal gates also pass every229376 outputs,
per-head dense float64 checks, first-row identity, and identical cross-arm bits.
Cache133.7784/private140.7044/L3139.3344us at matched Bc128/Br16/R256;
L37168updates/3584initializations. These are small validation-case timings,
not target performance: L3 remains slower than cache here. One-entry gate
active at this update; T1024 gates and T2048/T4096 targets queued. No target
multihead speedup claim yet. Status/REPORT remain authoritative.

**L3-06 T1024 acceptance update:** all11 pre-target gates PASS, including
causal depth1 and full917504-output T1024 checks. Hq14/Hkv2/d64 causal T1024:
inherited cacheBc5121013.2496us; additional cacheBc256987.8636us;
privateBc256928.0472us; L3Bc256898.3396us. L3 latency11.341% lower than
inherited cache,9.062% lower than additional Bc256 cache,3.201% lower than
matched private. Retain the stronger measured cache control when interpreting
the small-case benefit. All same-Bc bits exact and independent math/config/
copy/command/resource checks pass. This is T1024 validation evidence; target
T2048cache/private active, L3 queued; allT4096 queued. No target result yet.

### L3-07 — Analytical Qwen parameter selection (2026-09-28)

**D/P; structural analysis implemented, simulations pending original targets.**
User requested analytical selection instead of blind sweeps. FA-09 validated
loop/traffic counts but not latency; FA-29 applied fitted bounded search with
17.08% historical held-out error. L3-05 single-head tuning and L3-06 causal GQA
are inputs, not rerun as new mechanisms. New model enumerates causal padding,
per-core work under the frozen rotated schedule, private V claims and L3 output
storage. All12 currently accepted L3-06 cases match exact counts. Four new host
tests pass; derived runner run() AST identical to original acceptance function.

Sparse Qwen R256 timing anchors cannot identify R-dependent latency. Explicit
sensitivity scenarios rank near-neighbor geometries; scores are NOT predicted
latency or causal fractions. Current selected Br16/Bc256/R128 and Br16/Bc512/R256
are tested on both private and L3, with cache frozen. R512/Bc256 at T2048 has
25.7% max/mean core arithmetic imbalance and27.5% more busiest-core work versus
R256; deferred for the initial budget, not ruled out at T4096.

Separate waiting coordinator starts only after original Qwen targets succeed.
Four T1024 full gates, four T2048 cases, at most two T4096 winner transfers.
Reuse old controls/oracles, same ELF/model/finite resources. Exact every-output,
config/counter/occupancy gates retained; max2sim with load guards. Decision is
lower complete ROI vs inherited arm, then strongest private/L3/cache comparison.
No new accepted tuning result yet. [Plan](QWEN_ANALYTICAL_TUNING.md),
[analytical report](../../results/cacheflex_l3_spm/qwen_analytical_20260928/REPORT.md),
[coordinator](../../results/cacheflex_l3_spm/qwen_analytical_20260928/coordinator.json).

### AI-10 — Qwen SwiGLU analytical tuning on matched FA hardware (2026-09-28)

**D/P; prepared, runtime validation pending.** User authorized FFN tuning toward
a component-composed projected Qwen prefill speedup. AI-03/04 already studied
KC/mapping/combination on older shapes; AI-06 accepted full FP16/FP32 SwiGLU
D2048/F8192; AI-08 counted accumulator traffic but did not isolate its runtime;
AI-09 is an active finite-model transfer of that older shape. These are not
untested mechanisms. L3-06/07 supply the current Qwen FA profile/search.

New: D896/F4864, eight cores, VL16, exact lifecycle build_v5/finite FA platform;
existing guests copied byte-for-byte. Independent cache/SPM selection from exact
per-core loop/copy/accumulator accounting plus uncalibrated capacity sensitivity.
Equal four-case budgets at T1024, then inherited control plus each backend's
winner at T2048/T4096. Cache candidates MP/KC/combined:2/128/0,4/512/0,2/256/1,
8/512/0; SPM:2/128/0,4/512/0,4/512/1,8/1024/0. Scores are not predicted latency.
Four host tests pass, including Qwen tails/empty partitions and rejection of an
unrelated hardware change. No accepted Qwen FFN performance result yet.

Four compatibility gates and eight T24 Qwen-shape gates run one simulator at a
time; target search waits for FA analytical completion, then max2new simulators.
Full ordered oracle/float64-chain/export/exact-peer-output/config/copy/finite
occupancy checks. Full first-invocation ROI, including packing, is the tuning
objective; second call is a separately reported diagnostic, not a warm median.
Decision: lower full ROI versus inherited configuration and strongest measured
cache/SPM comparison. Preserve negative/regressive results. No new L3 FFN engine;
CF+L3 initially shares CF's FFN timing in the eventual projection. Remaining
projection/norm/RoPE/residual/logits and integration/packing policy work required.
[Plan](QWEN_SWIGLU_TUNING.md),
[selection](../../results/ai_workloads/qwen_swiglu_20260928/SELECTION.md),
[status](../../results/ai_workloads/qwen_swiglu_20260928/experiment_v1/status.json).


**L3-06 completion update:** all 17 cases accepted.
T1024 L3 898.3396us versus cache 1013.2496us: 11.341% lower latency (1.127914x).
T1024 L3 898.3396us versus private 928.0472us: 3.201% lower latency (1.033069x).
T2048 L3 2592.1504us versus cache 3303.0620us: 21.523% lower latency (1.274256x).
T2048 L3 2592.1504us versus private 2700.3324us: 4.006% lower latency (1.041734x).
T4096 L3 9053.6880us versus cache 10527.9640us: 14.003% lower latency (1.162837x).
T4096 L3 9053.6880us versus private 9473.3184us: 4.430% lower latency (1.046349x).
Full causal GQA oracle/exact output/configuration/resource checks; model-shaped FP16 kernel, not full BF16 inference.

### L3-08 — Add missing Qwen FA cache analytical tuning (2026-09-28)

**D/P; prepared and queued, no new cache result yet.** User explicitly supersedes
the earlier frozen-cache restriction. FA-06/29 previously tuned older baselines;
L3-05 included direct CPU PV epilogues. L3-06 supplied inherited Qwen controls;
L3-07 omitted cache from its two-candidate follow-up. New work applies the same
bounded structural/sensitivity methodology to cache and closes that comparison
gap, without editing the live SPM/L3 queue.

Selected cache R256/Bc256 (ranked) and R128/Bc256 (grouping diagnostic), Br16.
Including R256/Bc512 inherited control gives all three backends the same three
T1024/T2048 geometries and two-candidate budget. Reuse accepted T1024R256/Bc256;
one new T1024 gate, two T2048 cases, at most one improving T4096 transfer.
Same guest_v4/build_v5/finite profile, full oracle/exact bits/config/count/bounds.
AST audits preserve run/work/reference and phase acceptance, only worker limit
2->1 changes. Scores are not latency predictions. Decision is lowest accepted
complete ROI per backend, including old controls; no global optimum claim.

Queued now, simulations after L3-07 completes, one at a time, allowing existing
FFN and three older jobs within six total slots. No jobs stopped/inputs edited.
[Plan](QWEN_CACHE_TUNING.md),
[selection](../../results/cacheflex_l3_spm/qwen_cache_analytical_20260928/selection.json),
[coordinator](../../results/cacheflex_l3_spm/qwen_cache_analytical_20260928/coordinator.json).


**L3-07 completion update:** baseline plus selected follow-up, 25 cases accepted.
T1024 L3 800.4924us versus cache 987.8636us: 18.967% lower latency (1.234070x).
T1024 L3 800.4924us versus private 818.6768us: 2.221% lower latency (1.022717x).
T2048 L3 2592.1504us versus cache 3303.0620us: 21.523% lower latency (1.274256x).
T2048 L3 2592.1504us versus private 2700.3324us: 4.006% lower latency (1.041734x).
T4096 L3 9053.6880us versus cache 10527.9640us: 14.003% lower latency (1.162837x).
T4096 L3 9053.6880us versus private 9473.3184us: 4.430% lower latency (1.046349x).
Full causal GQA oracle/exact output/configuration/resource checks; model-shaped FP16 kernel, not full BF16 inference.

### L3-09 — Finite direct local core-to-L3 reduction link (2026-09-28)

**D/P; implemented, building; no accepted runtime result.** User authorized a
physical direct-path experiment on the best measured Qwen FA configuration.
L3-02/03 already implemented bank-local reduction and reduced NoC payload;
L3-04/05 improved accumulator lifecycle and PV epilogue; L3-06/07 tuned Qwen
private/L3 kernels; L3-08 cache tuning remains independent and active.
Materially new: bypass local Garnet transport for op7 commands/completions with
fixed core-i/bank-i links, finite serialization/credit/backpressure, sharing
existing protocol receive ports and bank bandwidth. Eight 128-bit directions
pairs at 2GHz, 2cy wire + endpoints, 2304B bounded data buffers, no new ALU or
accumulator SRAM. 64-bit/4cy sensitivity; explicit uncalibrated hardware model.
All ordinary coherence/slot claims/final reads remain on existing paths.

Decision: lower accepted complete FA ROI than same-configuration NoC reduction,
then compare strongest independently measured private/cache. Directed timing or
lower flits alone is not success. Exact default controls, mixed/directed/stress
checks, full independent math/exact bits/config/copy/count/occupancy gates
precede matched best-kernel T1024/2048/4096. One new simulator, six total,
resource guarded; larger targets wait for cache tuning completion. Preserve negative results.
[Design and validation plan](L3_DIRECT_LINK.md),
[evidence](../../results/cacheflex_l3_spm/direct_link_20260928).


**AI-10 gates accepted; targets queued:** 12 accepted cases; full output oracle, float64 chain, exact peer bits, config/copy/finite-resource gates. Authoritative [report](../../results/ai_workloads/qwen_swiglu_20260928/experiment_v1/REPORT.md). Small gates are not target results. First/second call policies remain separate.


**L3-08 completion update:** all-backend accepted comparison with cache tuning, 29 cases accepted.
T1024 L3 800.4924us versus cache 848.8716us: 5.699% lower latency (1.060437x).
T1024 L3 800.4924us versus private 818.6768us: 2.221% lower latency (1.022717x).
T2048 L3 2592.1504us versus cache 2960.4948us: 12.442% lower latency (1.142100x).
T2048 L3 2592.1504us versus private 2700.3324us: 4.006% lower latency (1.041734x).
T4096 L3 9053.6880us versus cache 10527.9640us: 14.003% lower latency (1.162837x).
T4096 L3 9053.6880us versus private 9473.3184us: 4.430% lower latency (1.046349x).
Full causal GQA oracle/exact output/configuration/resource checks; model-shaped FP16 kernel, not full BF16 inference.


**L3-11 completion update:** equal three-geometry T4096 comparison, 34 cases accepted.
T1024 L3 800.4924us versus cache 848.8716us: 5.699% lower latency (1.060437x).
T1024 L3 800.4924us versus private 818.6768us: 2.221% lower latency (1.022717x).
T2048 L3 2592.1504us versus cache 2960.4948us: 12.442% lower latency (1.142100x).
T2048 L3 2592.1504us versus private 2700.3324us: 4.006% lower latency (1.041734x).
T4096 L3 9053.6880us versus cache 10509.4604us: 13.852% lower latency (1.160793x).
T4096 L3 9053.6880us versus private 9473.3184us: 4.430% lower latency (1.046349x).
Full causal GQA oracle/exact output/configuration/resource checks; model-shaped FP16 kernel, not full BF16 inference.


**L3-12 completion update:** bounded adaptive R/Bc neighbor comparison, 46 cases accepted.
T1024 L3 800.4924us versus cache 848.8716us: 5.699% lower latency (1.060437x).
T1024 L3 800.4924us versus private 818.6768us: 2.221% lower latency (1.022717x).
T2048 L3 2592.1504us versus cache 2960.4948us: 12.442% lower latency (1.142100x).
T2048 L3 2592.1504us versus private 2700.3324us: 4.006% lower latency (1.041734x).
T4096 L3 9053.6880us versus cache 10509.4604us: 13.852% lower latency (1.160793x).
T4096 L3 9053.6880us versus private 9473.3184us: 4.430% lower latency (1.046349x).
Full causal GQA oracle/exact output/configuration/resource checks; model-shaped FP16 kernel, not full BF16 inference.

## Deliberation template and maintenance

Before a new experiment, write:

> Related ledger IDs: … . Previously tested: … . Outcome and limitation: … . The new hypothesis differs by … . A result that would change our decision is … . We will measure matched end-to-end ROI and verify … .

New entry template:

> **ID / date / mechanism.** Grade N/D/H/X/P. Workload, T/head count/core count/VL/precision, model revision/cache geometry and ROI/warm policy. Exact control/intervention timings and ratio. Validation evidence and source paths. Key takeaway. Confounds/limits. Relationship to earlier entry (confirms, supersedes, contradicts, or transfers). Unfinished cases explicitly separated.

Refresh: `python3 scripts/index_experiment_history.py`. Do not edit old accepted results to update this ledger. Never replace a previous result just because its successor is faster; keep both and explain the changed assumptions.


**L3-09 validation update:** build_v4/experiment_v2 all13 runtime gates PASS,
including3 exact default controls,2 enabled-noop exact controls,6 directed
ordering/arithmetic/stress cases and2 full Qwen T256 checks. All31 sourceguards
pass. Nominal T256139.3344->138.1844us (0.825%lower), depth1/64bit/4cy stress
156.8424->148.6124us; these are validation workloads, not best-kernel targets.
Direct command/ACK/byte/beat counts exact, finite pending<=1, full oracle/exact
bits/configuration/resource bounds. T1024 best kernel pair now active; larger
sizes queued. Earlier compile/superseded attempts retained separately.


**L3-09 first best-kernel FA acceptance:** Qwen T1024 Hq14/Hkv2/d64, 8 cores,
Br16/Bc256/R128, same new ELF: NoC800.4924us, direct795.9964us, **0.56165%
lower complete ROI (1.005648x)**. All917504 outputs checked against independent
oracle and exact prior bits; configuration, command/copy counts and finite
bounds pass.35840 direct commands and ACKs;5734400 data/header bytes over358400
beats. NoC injectedflits3143596->2786313(-11.365%); meancommandcycles
352.7104->306.1316(-13.205%). These overlapping counters are not additive ROI
fractions. Transport accelerates commands substantially more than overall FA;
this conservative onecredit/source-FIFO-retaining design is only a small net gain.
Same-new-ELF private/cache and slowlink repeats plus T2048/T4096 remain pending;
no conclusion about their results or all possible dedicated-link implementations.
Accepted evidence: direct_link_20260928/experiment_v2/runs/t1024_{noc,direct};
automatic accepted-only EFFECTS.md/diagnostics.json via analyze_l3_direct.py.


### L3-10 — Qwen causal FA long-context transfer, T16K/T32K (2026-09-28)

**P; queued, coordinator503531 waiting for L3-09.** FA-28/29/31 tested older
single-head T16K and found traffic savings did not guarantee best-to-best wins;
L3-06/07/08 tuned Qwen through4096; L3-09 tests direct transport. New case is
14Q/2KV causal GQA at16384 and32768 on the same finite build_v4, four backends
(cache/private/NoC-reduction/direct-reduction), inherited best4096 tiles, not
independent long-context tuning. Same physical caches:64KiB I+D each/core,
512KiB L2/core,4MiB LLC total. No added SRAM or changed timing.

Isolated guest lifts only16K admission guard to32K; all .text except one MOV
immediate and all kernel headers match. Four exact T1024 old/new-guest controls
precede eight targets, after L3-09 finishes. Full independent oracle every
output, separateBc references, exact sameBc peers, counts/config/finitebounds.
Single-thread chunked references; max1new/six total sims, resource guards,
96h per-job cap for long simulations/oracles. Decision is complete FA ROI
speedup against cache/private and matched NoC/direct, retaining negative cases.
[Plan](QWEN_LONG_CONTEXT.md),
[queue root](../../results/cacheflex_l3_spm/qwen_long_context_20260928).

L3-10 launch:25 host preflight checks PASS (command changes limited to guest/T/outdir, all existing control epoch counts, missing-command rejection, closed-form long causal work counts). Coordinator503531 live; experiment_v1/status.json has4 exact controls +8 targets pending. New guest runtime gates and long-context results are NOT yet accepted. Six existing simulators left untouched; host92%idle/678GiB available at preflight.

### AI-11 — Qwen QKV/output projection tuning (2026-09-28)

**P; user authorized, preparing.** AI-01/04 already explored projection proxies;
AI-03 exposed packing confounds; AI-06/10 provide accepted FP32 GEMM primitives.
New Qwen fusedQKV896->1152 (bias, Q14/K2/V2 head layout) and output896->896
(no bias, head-to-token conversion), current finite FA/FFN platform. Reuses
unchanged GEMM/staging code; new driver with full ordered+float64 references.
Equal4candidate MP/KC budget per operator/backend, T24/25 gates thenT1024
screen and inherited+winner T2048/T4096. Full first ROI decides benefit;
second call separate. No accepted runtime yet; no L3 projection offload.
[Plan](QWEN_PROJECTION_TUNING.md). Max1new/six total, resource guarded.

### MOD-09 — Pre-RTL hardware feasibility and headroom audit (2026-09-28)

**D; user authorized after projection queue launch.** L3-02/03/04/09 already
implemented reduction, bank pipelining, initialization/finalization and direct
transport; MOD-04 documents memory-model gaps, FA-29 warns against inferring
runtime from traffic/stall sums. New work audits current frozen implementation
against byte/lane/queue/port bounds, inventory of storage/wiring and clocks,
and accepted complete-ROI ablations. No new simulator sweep initially.
Decision: identify supported assumptions, contradictions, uncalibrated costs,
and evidence-supported priorities before RTL or optimization investment.
Any future sensitivity/oracle runs require separate recorded configurations;
analytic bounds are not measured FA savings or physical timing closure.

**AI-11 launch update:** guest_v2/experiment_v2 coordinator570871 live,
resource-waiting at first T25 QKV cache gate. Max1new/six total; no old jobs
interrupted. Five host tests PASS: exact core arithmetic/staging preservation,
causal-independent projection row/head layouts, ordered+float64 references and
corruption rejection, MP/KC/tail work counts, full configuration isolation.
V1 coordinator564139 stopped before any simulation to remove an unused QKV
FP16 conversion/store; all v1 evidence retained. V2 skips only that epilogue,
then FP32 bias + final FP16/head-layout conversion; same change in both arms.
No accepted projection runtime yet. Authoritative plan
[QWEN_PROJECTION_TUNING_V2.md](QWEN_PROJECTION_TUNING_V2.md),
status results/ai_workloads/qwen_projection_20260928/experiment_v2/status.json.

**MOD-09 diagnostic completed:** frozen direct build_v4 source/config/accepted
records audited. No arithmetic/byte-service impossibility identified; physical
2GHz clock, bank macro/ECC/arbitration, area/energy/wiring remain uncalibrated.
Nominal one-credit forward command limit15Ruby cycles=19.2GB/s command bytes,
not32GB/s sustained raw line rate. C++ flight records retain256B contributions;
metadata-only physical storage requires ownership proof. Bank serialized normal
start-to-next-start floor54cycles, not85:35cycle response queued after install.
AtT2048direct busiest bank compulsory reduction SRAMport demand2.456% ofROI,
ALUissue1.367%, serialized schedule floor16.025%, rawlinkbeats3.075%.
These exclude other traffic/queues and ARE NOT measured utilization or removable
runtime fractions. No new sensitivity/oracle simulation or RTL performed.
[Audit](../../results/cacheflex_l3_spm/hardware_audit_20260928/REPORT.md)
with source/artifact hashes and machine-readable calculations. Hardware width/
ALU expansion is not yet justified; source/credit/bank-latency criticality needs
isolated full-ROI evidence before investment. Prior pipeline/finalization/retry
experiments retained as tested mechanisms, not rediscovered proposals.

AI-11 first runtime gate now active: T25 QKVcache simulator571181; coordinator570871.
Initial invocation guest PASS errors0/exact reference; full acceptance pending
second invocation and host/config/resource checks. Six total simulators maintained.

MOD-09 refreshed with acceptedT4096direct9037.8088us vsNoC9053.6880us:
0.175389% lower fullROI; all3670016 outputs fulloracle/exactbits/config/count/bounds.
Remaining current-build cache/private/slow repeats pending. Busiest bankT4096
ALUissue floor1.416%/SRAMport2.674% ofROI; not measured utilization or runtime shares.
Audit additionally derives conditional8/4lane arithmetic/conversion/II schedules
(13/10/8 and21/18/16cycles) as narrower-engine hypotheses. These are NOT measured
latencies/area/speedups; decision would be whether smaller-engine fullROI retains
benefit over privateSPM. No new sensitivity simulation queued by this audit.

AI-11 first runtime gate ACCEPTED: tail_gates_qkv_cache_t25_mp2_kc128, first313.9256us/second97.2572us; every28800 output bits match ordered FP32+bias oracle, float64 check/config/copies/finitebounds PASS. Small validation timing only; no target speedup yet. Coordinator continues automatically.

**AI-10/11 composition preparation; MOD-09 follow-up design:** accepted-only
collector scripts/collect_qwen_components.py now refreshes once/minute without
launching simulators. Four accounting tests PASS: reject altered/missing acceptance,
exclude partial/gate timings, select first-call rather than warm-call winners,
reuse private-SPM FFN for L3 arms without inventing missing operator costs.
Source records/hashes retained; layer/model totals remain null pending remaining
operators, activation integration and cache-state/build equivalence validation.
First accepted AI-10 target: T1024 cache MP4/KC512 first30107.144us; search still
running, no accepted SPM target comparison at this entry. This is a best-so-far
component measurement, not end-to-end inference.
[Accounting](../../results/ai_workloads/qwen_composition_20260928/REPORT.md),
[boundaries and diagnostic design](QWEN_COMPOSITION_AND_HEADROOM_NEXT.md).
Narrow8/4lane engines, source/completion traces and separated bank-delay
sensitivities are documented proposals only, NOT implemented or queued here.
Each cites prior mechanisms/model deltas and a complete-FA-ROI decision criterion.


**L3-09 completion update:** all default/directed/full FA gates accepted.
T1024 direct 795.9964us vs noc 800.4924us: 0.562% lower latency (1.005648x).
T1024 direct 795.9964us vs private 818.6768us: 2.770% lower latency (1.028493x).
T1024 direct 795.9964us vs cache 848.8716us: 6.229% lower latency (1.066426x).
T2048 direct 2584.7648us vs noc 2592.1504us: 0.285% lower latency (1.002857x).
T2048 direct 2584.7648us vs private 2700.3324us: 4.280% lower latency (1.044711x).
T2048 direct 2584.7648us vs cache 2960.4948us: 12.691% lower latency (1.145363x).
T4096 direct 9037.8088us vs noc 9053.6880us: 0.175% lower latency (1.001757x).
T4096 direct 9037.8088us vs private 9473.3184us: 4.597% lower latency (1.048188x).
T4096 direct 9037.8088us vs cache 10527.9640us: 14.154% lower latency (1.164880x).
Finite transport assumptions and unchanged bank/ALU resources; full math/exact bits/config/count/bounds. Sensitivity and all regressions retained in direct_link_20260928/experiment_v1/REPORT.md. No RTL-calibrated area/timing or full-model claim.

**AI-11 gates accepted; targets follow.** 20 accepted cases. Full ordered bits, float64 reference, peer hashes, config/count/resource checks. Report: results/ai_workloads/qwen_projection_20260928/experiment_v2/REPORT.md.

**AI-10/11 T1024 screens complete (2026-09-29 status audit):** all eight
AI-10 target candidates and all sixteen AI-11 target candidates accepted;
status records rechecked against accepted.json. First-call best-of-four per
backend: FFN cache MP8/KC512 26616.9476us vs SPM MP8/KC1024
19592.0280us (1.358560x); QKV cache MP8/KC512 2674.2280us vs SPM
MP8/KC1024 2051.0080us (1.303860x); output cache MP8/KC512
1858.6080us vs SPM MP8/KC1024 1419.7496us (1.309110x).
These are independently selected winners within the recorded analytical
candidate budgets, not exhaustive optima or integrated-model performance.
First-use packing/staging/layout policy retained; warm-call times separate.
Both queues now target T2048, T4096 follows. QKV T2048 cache inherited/winner
accepted (10034.2212/5080.6236us); corresponding SPM comparison pending.
L3-10 passed all four exact-guest gates; T16384 cache active, other long arms
queued. No accepted T16K/T32K Qwen timing at this audit.

### MOD-10 — Isolated Qwen L3 latency/engine diagnostics (2026-09-29)

**P; user explicitly authorized setup and queue, resource bounded.** L3-02
already implemented retry isolation; L3-03 pipeline was near-null on older
single-head kernels; L3-04 initialization helped/finalization regressed; L3-09
direct transport saves only0.175% at4096; MOD-09 audited costs. Material delta:
separate reduction-only read and post-install reply latency on current frozen
Qwen Br16/Bc256/R256, passive bounded source/flight trace, modeled8/4lane timing,
and revalidation of existing pipelining on current Qwen+init+direct.

New root results/cacheflex_l3_spm/diagnostics_20260929. Source originals retained
under preserved/. build_v1 failed before compilation on SLICC Cycles/int compare;
fixed to intToCycles(0); failed source/log retained. build_v2 coordinator2698842
resource-waits, four low-priority compile workers. Standalone corrected SLICC
generation PASS; all37 host/source tests PASS (6new +31existing).
Queue scripts/run_l3_diagnostics.py experiment_v1 coordinator2710282 waits for
build_v2, then19 exact-default/no-op/directed/full-small gates before8T4096
targets: exact direct trace replay; reply1/read1/both1-cycle optimistic cases;
8/4lane timing; existing pipeline; NoC both-fast. No runtime gate accepted yet.

Default-zero overrides inherit old35cycle timing. Ordinary coherent/SPM accesses,
finalization, read/write port charges, installation-before-ACK, source/sink
capacities, scoreboard/barriers and numerical order remain unchanged. Trace
must preserve all numeric nonhost stats/output bits; no extra timed events.
All FA cases require every-output independent oracle, exact prior bits, config,
copy/command counts and finite occupancy checks. Failures stop downstream.
Decision: complete FA ROI versus identical source and strongest private control;
preserve negative/null results, never infer runtime from retries/lifetimes.
Optimistic delays are NOT feasible hardware points; narrower timing is NOT area
evidence. No new FFN reduction kernel is implemented by this experiment.

At most1new/six total sims, >=50%CPU idle/128GiB available RAM/100GiB disk/zero
swap admission; own-job pause on over-cap or sustained low idle; runtime64GiB
RAM/75GiB disk and96h limits. Existing production simulations untouched.
[Plan](L3_DIAGNOSTICS_20260929.md),
[status/report](../../results/cacheflex_l3_spm/diagnostics_20260929/experiment_v1/REPORT.md).

**MOD-10 build/default update (2026-09-29):** build_v2 complete SHA95105acb1d108ed317c21840ca9c301d2d69d6f325aa98a6a4e16ee5ff6a23a5. Four new-binary exact small controls accepted (cache133.7784/private140.7044/NoC139.3344/direct138.1844us), all prior finite numeric nonhost stats/output bits exact. Trace-equivalence case active PID3897954 at check, then remaining gates and8T4096targets. No accepted diagnostic intervention/performance result yet; coordinator2710282 live. Six total simulators,47%host idle,366GiB available,zero swap at check.

**MOD-10 runtime gates accepted:** 19 default/no-op/directed/full small-Qwen cases; full oracle/exact bits/config/count/bounds. Targets follow. See /data2/rsappidi/cacheflex_mc/results/cacheflex_l3_spm/diagnostics_20260929/experiment_v1/REPORT.md.

### OPS-20 — Shared 20% CPU/RAM scheduling budget (2026-09-29)

User explicitly authorized parallel execution of existing experiments under
20% CPU and RAM limits, superseding six-simulator scheduling limits. Related
MOD-10 (19 gates and exact T4096 trace accepted), L3-10 (four gates and T16K
cache accepted), AI-10 and AI-11 (gates/T1024 selection complete). This is a
scheduling-only continuation, no new scientific configurations. Existing live
simulators are adopted without restarting; old coordinators are SIGSTOPped
until their children are accepted, then retired to prevent duplicate launches.
Independent T2048/T4096 component targets use already frozen T1024 winners.
Original full arithmetic/config/count/finite-resource acceptance tails are
executed unmodified; four completed-run replays passed. Cross-worker peer
output-bit comparison is repeated before authoritative acceptance.
Budget: 128 logical CPUs and 810606362624 B RAM; experiment affinity uses
24 logical CPUs (18.75%). Each job reserves 6 GiB: 1.5 GiB worker and 4.5 GiB
simulator enforced with reversible RLIMIT_AS soft limits. Existing llama
coordinator/simulator reserve another 6 GiB; 8 GiB outside-work margin.
20% RAM limit=162121272524 B (~150.987 GiB), allowing22 worker slots while
llama is active and up to23 when released. All-UID RSS/CPU monitored; no
privileged cgroup delegation, so unrelated future external programs cannot
be hard-capped by this scheduler. No other user's processes are changed.
Complete simulated ROIs and acceptance criteria unchanged; host wall time is
not a performance result. Journal/status: results/parallel_budget_20260929.

**OPS-20 launched/verified:** scheduler992107, all5 original active simulator PIDs preserved; four original coordinators intentionally stopped for adoption.22worker slots plus existing llama; resource sample CPU17.9% of128 logical CPUs, userRSS2.5% RAM. All frozen manifests rechecked: {"diagnostics": 312, "long": 310, "ffn": 515, "projection": 509}. All worker/simulator affinity and allocation limits checked; no active failures. Scheduler automatically publishes original-check accepted results to existing reports.

### AI-12 — Remaining Qwen pipeline component coverage (2026-09-29)

**P; user authorized; built and host-validated, runtime gates pending.** Related
AI-10/11 already cover FFN/projections including SiLU, QKV bias/head layout and
output head merge; AI-03/04 distinguish first-use from prepacked policy; L3-06..10
cover FA; OPS-20 governs shared resource use. Inventory search found no prior
RMSNorm/rotary/embedding/logits study. New seven-operator suite: RMSNorm,
residual, RoPE application, shared rotary tables, embedding, final RMSNorm,
last-position vocabulary GEMV. D896, Q14/KV2,24layers,vocab151936; synthetic
FP16/FP32. Actual SVE2048 guest, same ELF both platforms; true single-vector
logits with native tied-weight layout, no padded GEMM or uncharged transpose.

Analytical stream/reuse/register accounting selects equal three-case budgets:
row blocks1/8/32, logits live dot blocks1/2/4. Coherent SVE fallback is available
on the private-SPM platform because these operands have little reuse; zero SPM
copies/reductions. This is not a new SPM arithmetic engine or exhaustive proof
that alternative SPM kernels cannot help. Independently select each platform's
first-complete-ROI winner at T1024; transfer winner plus block1 to T2048/T4096.
Full-vocabulary last-token logits measured once; all-token logits/decode excluded.

T1/V17 and T25/V263 runtime gates precede targets. Full independent output
oracle, exact peer/repeat bits, export hash, hardware whitelist, finite bounds
and zero-command checks; errors stop downstream/peers. All7 host tests passed;
full-vocabulary reference generation verified under1.5GiB allocation limit
(peak RSS578528KiB). Compiled guest SHA141b98c6feb54d237930252807462686642a84062e3f995d50c32fe9ec0f71a3.
No runtime/timing acceptance yet. First invocation remains the objective, second
call diagnostic. Decision is lower full ROI versus block1 and best measured
matched platform, not traffic/stall sums. Integration/cache-state validation
still required; no model latency total or paper-ready claim from preparation.

New queue reserves maximum future concurrency of the unchanged old plan, then
fills released slots under shared24-CPU affinity and20%RAM budget. It does not
restart/resume old coordinators. At most22 total new slots when old plan drains,
6GiB reserved per admission, existing llama and outside margin retained.
[Plan](QWEN_REMAINDER_TUNING.md), [status](../../results/ai_workloads/qwen_remainder_20260929/experiment_v1/status.json).

**AI-12 launch verified:** coordinator3018928 live, zero new simulators while
existing queue holds the budget. Automatic admission/gates follow as slots free.
Affinity24CPUs and1.5GiB coordinator limit verified; all old four manifests and
new483-entry manifest match hashes. Fresh sample userCPU17.8%, RAM3.6%.

**Parallel-budget completion diagnostics:** All 27 cases accepted using unmodified original acceptance checks. Effects [{"name": "t4096_direct_trace", "roi_us": 9037.8088, "source_us": 9037.8088, "latency_reduction_pct": 0.0, "speedup_vs_private": 1.048187520851293, "speedup_vs_cache": 1.1648801421866768}, {"name": "t4096_direct_reply_fast", "roi_us": 9032.074, "source_us": 9037.8088, "latency_reduction_pct": 0.06345343353579835, "speedup_vs_private": 1.0488530541268815, "speedup_vs_cache": 1.165619767951414}, {"name": "t4096_direct_both_fast", "roi_us": 9047.3656, "source_us": 9037.8088, "latency_reduction_pct": -0.10574244500500818, "speedup_vs_private": 1.04708031252766, "speedup_vs_cache": 1.1636496705737194}, {"name": "t4096_direct_lanes4", "roi_us": 9034.2944, "source_us": 9037.8088, "latency_reduction_pct": 0.038885531634613724, "speedup_vs_private": 1.048595272697777, "speedup_vs_cache": 1.1653332882311207}, {"name": "t4096_direct_read_fast", "roi_us": 9030.098, "source_us": 9037.8088, "latency_reduction_pct": 0.08531714014574643, "speedup_vs_private": 1.0490825680961602, "speedup_vs_cache": 1.1658748332520865}, {"name": "t4096_noc_both_fast", "roi_us": 9074.1244, "source_us": 9053.688, "latency_reduction_pct": -0.22572458869800638, "speedup_vs_private": 1.0439925641751175, "speedup_vs_cache": 1.1602181693695977}, {"name": "t4096_direct_pipeline", "roi_us": 9037.1788, "source_us": 9037.8088, "latency_reduction_pct": 0.006970716176257774, "speedup_vs_private": 1.0482605921219574, "speedup_vs_cache": 1.1649613483358325}, {"name": "t4096_direct_lanes8", "roi_us": 9039.4244, "source_us": 9037.8088, "latency_reduction_pct": -0.017876014371975835, "speedup_vs_private": 1.0480001801884642, "speedup_vs_cache": 1.1646719452623555}]. Scheduling-only continuation: results/parallel_budget_20260929.

**MOD-10 completed trace/PC analysis (2026-09-29):** Read-only analysis of all
accepted diagnostic ROIs, exact frozen guest disassembly, full-run PC counters,
and joined per-core command traces. No new simulator launch/model change.
Read-fast mean command lifetime349.85->178.01 corecycles (-49.117%), source-full
retries30,967,717->16,148,444 (-47.854%), but full FA only0.0853% faster.
Both-fast lifetime143.93 (-58.860%), retries13,085,810 (-57.744%), yet0.1057%
slower. Four-lane lifetime496.55 (+41.935%), retries43,474,385 (+40.386%), with
FA effectively unchanged (-0.0389%). Pipeline lifetime228.72 (-34.623%) and
retries19,523,304 (-36.956%) buy only0.0070%. These demonstrate substantial
queue/lifetime changes hidden by other work; counters are not additive runtime.
Full-run aggregate ROB-head residence /8-core elapsed cycles: QK main19.079%,
PV loop21.496%, softmax main31.684%, P packing4.779%, separate scalar exp3.623%.
These are verified code ranges, NOT serial phase times/removable fractions.
Reduction submission epilogue0.325%; posted stores can move apparent blockage
later, so that small share alone does not prove capture harmless. Regular
requests accepted while reduction blocked12,696,138 confirm isolation progress.
New coverage caveat:100us trace starts5.772us before ROI; first capture at
ROI+88.1636us, leaving only6.0644us reduction activity.275/487424commands
(0.05642%),235completed/40right-censored, all serials0..43/core in the initial
256-row contribution pass. Thus startup initialization only, no steady-state
normal accumulator reads or terminal drain. Do not generalize conditional
full-queue samples to full FA or use them as evidence of dominant source stalls.
The incremental direct-L3 gain over private remains4.597% lower latency;
NoC->direct0.175%. Prior L3-02 retry isolation and L3-05 direct PV epilogue already
addressed those mechanisms. CPU-side operand delivery/softmax/P packing are
remaining candidates, not proven removable headroom. Any source-criticality
follow-up needs steady-state/drain coverage and commit/issue correlation with
exact instrumentation gates; no such new run launched. Report and reproducible
analysis: [trace analysis](../../results/cacheflex_l3_spm/diagnostics_20260929/trace_analysis_v1/REPORT.md),
`scripts/analyze_l3_diagnostic_traces.py`. Source hashes and raw derived records
retained in inputs.json/analysis.json. No hardware area/clock claim.

**MOD-10 follow-up: PV-extension analytical screen (2026-09-29).** User asked
about expanding arithmetic coverage. Prior L3-02/03 recurrence/FP16 transport,
L3-05 direct epilogues, MOD-09/10 timing diagnostics are existing mechanisms;
no accepted near-L3 PV matrix-product engine found in ledger/inventory. New
candidate moves PV plus recurrence, about50.10% of QK/PV/update FLOPs atBc256,
not50% of runtime/all attention work. Busiest T4096 bank64000updates requires
1,048,576,000 scalar PV FMAs. At2GHz and ideal oneFMA/lane/cycle,16/32/64/128/256
lanes require at least32.768/16.384/8.192/4.096/2.048ms versus9.0378ms fullFA.
Current16FP32 mul/add lanes are not native FP16FMA lanes; bounds are optimistic.
Br16 P payload8192B versuscurrent2048B contribution (4x), plus placement/fill of
32768B V now resident in privateL2, finite staging and shared64B/cycle bank port.
FP16 FMA accumulation order and FP32 nonfused recurrence must remain matched.
No simulator implementation/queue or performance claim. Next decision is finite
operand-reuse/schedule feasibility, then lower fullROI versus independently tuned
private/cache if a candidate passes. [Design](L3_PV_EXTENSION_FEASIBILITY.md).

**Parallel-budget completion projection:** All 52 cases accepted using unmodified original acceptance checks. Effects [{"t": 1024, "op": "qkv", "cache": "screen_t1024_qkv_cache_t1024_mp8_kc512", "spm": "screen_t1024_qkv_spm_t1024_mp8_kc1024", "cache_us": 2674.228, "spm_us": 2051.008, "speedup": 1.3038603457421913}, {"t": 1024, "op": "out", "cache": "screen_t1024_out_cache_t1024_mp8_kc512", "spm": "screen_t1024_out_spm_t1024_mp8_kc1024", "cache_us": 1858.608, "spm_us": 1419.7496, "speedup": 1.3091097190659535}, {"t": 2048, "op": "qkv", "cache": "target_t2048_qkv_cache_t2048_mp8_kc512", "spm": "target_t2048_qkv_spm_t2048_mp8_kc1024", "cache_us": 5080.6236, "spm_us": 4562.4856, "speedup": 1.1135648515800247}, {"t": 2048, "op": "out", "cache": "target_t2048_out_cache_t2048_mp8_kc512", "spm": "target_t2048_out_spm_t2048_mp8_kc1024", "cache_us": 3631.5244, "spm_us": 3349.3648, "speedup": 1.084242719694194}, {"t": 4096, "op": "qkv", "cache": "target_t4096_qkv_cache_t4096_mp8_kc512", "spm": "target_t4096_qkv_spm_t4096_mp8_kc1024", "cache_us": 11259.8416, "spm_us": 9070.3012, "speedup": 1.2413966583601435}, {"t": 4096, "op": "out", "cache": "target_t4096_out_cache_t4096_mp8_kc512", "spm": "target_t4096_out_spm_t4096_mp8_kc1024", "cache_us": 8399.9668, "spm_us": 6758.5572, "speedup": 1.2428639059235898}]. Scheduling-only continuation: results/parallel_budget_20260929.

### L3-11 — Complete equal-candidate Qwen T4096 tuning (2026-09-29)

**P; user authorized, prepared.** L3-06/07/08 tuned three common geometries at
T1024/T2048, but only inherited/selected transfers at T4096. Cache smaller-size
winner11015.6924us loses to inherited10527.9640us there. New work fills exactly
five missing T4096 cells: cacheR256Bc256; private/L3 eachR128Bc256 andR256Bc512.
Br16, same guest_v4/lifecycle build_v5, NoC FP16+init L3, no hardware changes.
Reuse four accepted controls and both full T4096 independent references. New
target_structure.json records target-size analytical counts/footprints.

AST identity verifies command construction, whole original acceptance suffix,
work/label/key/reference functions. Three preflight tests pass (300 future-budget
scenarios, no duplicates/equal geometry coverage, original acceptance identity).
Full-output oracle/exact peer bits/config/count/finite-resource gates unchanged.
Select best complete ROI per backend; preserve null/regressive results. This is
equal bounded target-size tuning, not proof of global optimum. No additional
neighbor sweep or direct-link transfer launched.

Shared24CPU affinity and20%RAM ceiling; at most4new sims with8GiB AS each,2GiB
coordinator. Reserve ALL old outstanding future slots, including paused families,
at6GiB each, plus llama6GiB/outside8GiB. Current old16 slots leaves4new slots.
Old long/FFN host-allocation failures and paused peers are preserved unchanged;
AI-12 stopped after59accepted cases on prior-queue failure guard. No accepted
T4096 control was invalidated by those unrelated failures.
[Plan](QWEN_T4096_TUNING.md),
[queue](../../results/cacheflex_l3_spm/qwen_t4096_tuning_20260929/experiment_v1/status.json).

**L3-11 launch verified:** coordinator3676105, four active simulators/one waiting. Actual24CPU affinity,8GiB AS, simulation entry and no startup errors verified for every child. Current all-user CPU3.93%, RAM3.02%; no pressure. No new accepted target result yet. Old queues untouched.

### OPS-21 — Recover host AS-limit failures without exceeding20% (2026-09-29)

**P; user authorized.** OPS-20 introduced4.5GiB simulator RLIMIT_AS; actual
T16K NoC andT2048SwiGLUmp8kc1024 host allocations failed, pausing12peers.
New scheduling-only supervisor adopts peers with unchanged PIDs/state, archives
failed attempts with hashes, restarts only2terminated cases, raises admitted
simulator/llama AS to8GiB. Original frozen sources/acceptance unchanged; worker
AST differs only2host-limit constants. CPU24/128; reserve full8GiB sim+1.5GiB
worker,8GiB outside,9.5GiB llama,1GiB supervisor,34GiB potentialL3-11. Initial
2retries+2peer upgrades yields150.5GiB<150.987GiB(20%); other peers queued until
reservations free. All-user CPU/RSS monitored; failures stay fail-closed.
Four old manifests verified (318/310/515/515files), both failed dirs unaccepted.
AI-12 experiment_memory_v2 auto-continues after recovery finishes, reusing59
accepted cases, unchanged scientific functions checked by AST. Old failure
evidence remains. [Plan](PARALLEL_MEMORY_RECOVERY.md).

**OPS-21 launch verified:** supervisor3844641, old992107 stopped. All12preserved simulator IDs retained; T16Kprivate3693174/direct998079 resumed at8GiB; exact-command failed retries T16KNoC3844663 andT2048SwiGLU3844669 running. Remaining10peers budget-queued, no restart. Llama27574 also8GiB. Combined reservation150.5GiB<150.987GiB; actual CPU7.03%, RAM3.35%. No new failures. AI-12 continuation automatically follows recovery targets,59cases reused.

**L3-11 partial acceptance: four of five new T4096 cases** pass original full-output/config/count/resource checks. CacheR256Bc25610509.4604us improves inherited10527.9640us by0.17576%; privateR256Bc51210255.7868us loses to9473.3184us control. L3R128Bc2569322.9480us andR256Bc5129873.1084us both lose to9053.6880us R256Bc256 control. PrivateR128Bc256 still running; do not call target-size search complete or its final private denominator selected. Report: results/cacheflex_l3_spm/qwen_t4096_tuning_20260929/experiment_v1/REPORT.md.

### L3-12 — Adaptive target-size R/Bc neighbor check (2026-09-29)

**P; user authorized parallel continuation.** Related FA-29,L3-07/08/11: current
three-geometry target-size search suggestsR256Bc256, but R512/Bc256 andR256/Bc128
remain unmeasured. First two geometries tested on cache/private/NoC-L3, Br16.
Model R512 halves staging nearly but raises max work10.4%; Bc128 reduces
footprint and doubles L3 commands. Full T1024 gates precede T4096 for each new
geometry. Same guest_v4/build_v5, exact original run/reference/work ASTs.

Adaptive union of each backend winner's unmeasured axis neighbors, same screened
R{64,128,256,512}/Bc{128,256,512,1024} grid, fixedBr16. Maximum6new geometries
(18targets+18gates); if frontier remains at budget exhaustion, report incomplete
coverage, not convergence. Import final L3-11 private result before adaptation.
Best validated complete ROI is decision criterion; retain null/regressive cases.

Parallel2new sims,8GiB each+2GiB coordinator. State-preserving recovery-supervisor
handoff introduces18GiB reservation; oldL3-11 reservation34->10GiB justified by
its frozen five-case plan with4accepted/1remaining. Original recovery admission/
publication loop unchanged; all simulation PIDs retained.24CPU/20%RAM limits.
Preflight original run AST, analytical frontier/direction, budget boundaries and
compilation pass. [Plan](QWEN_NEIGHBOR_TUNING.md).

**L3-12 launch verified:** coordinator4021707; initial cache/privateT1024R512Bc256 gates4021900/4021928 entered simulation,8GiB AS/24CPU each, no errors. Supervisor4021249 preserves all14recovery pairs; old3844641/992107 remain intentionally stopped. Total reservations148GiB<150.987GiB; actualCPU7.03%, RAM3.97%. No accepted neighbor timing yet.

### OPS-22 — Context-sized host allowance and proactive growth (2026-09-30)

**P; user authorized.** OPS-21's8GiB AS still insufficient: T16Kdirect failed
host bad_alloc; no simulated-architecture result. New scheduler-only recovery
preserves8live pairs, archives failed direct attempt with hashes, restarts only
that case. T16K allowance24GiB, T32K48GiB on admission; FFN/llama unchanged8GiB.
At2GiB below AS ceiling, grow by8GiB only if total reservations fit20% RAM;
otherwise pause the affected process until quota is available. CPU affinity24/128
unchanged. Require available server RAM minus full remaining user allowance to
leave at least64GiB headroom. Preflight available579GiB, projected spare448GiB.
All3T16K can run with2FFN/four preservedT32K at141.5GiB reserved<150.987GiB cap.
Original worker changes onlyhost limit parameters; original scientific acceptance
and publication block unchanged, all old manifests verified. Frozen simulator,
guest, config, inputs and successful results retained. Tests check limits/growth/
headroom/compilation. Root results/long_memory_recovery_20260930.

**OPS-22 launch verified:** supervisor3771764;8pairs/PIDs preserved; T16Kprivate3693174/NoC3844663 resumed24GiB, direct3771784 exact-command retry24GiB. FourT32K queued48GiB on admission. Archivehashes/command/affinity/limits checked; no new errors. Reservations141.5GiB<150.987GiB; actualCPU4.69%, RAM2.65%; projected spare server RAM414.6GiB. Proactive8GiB quota growth active.

### MOD-11 — Finite PV operand-schedule diagnostic (2026-09-30)

**D; completed analytical only, user authorized.** Related L3-02/03/05 already
implement recurrence/retry isolation/compact transport/direct epilogue; MOD-10
measured near-null sensitivity to recurrence acceleration and supplied PV FLOP
bounds; L3-11/12 completed tuned T4096 controls. New dimension is finite P/V
operand delivery and dependent FP16 FMA scheduling, not another recurrence sweep.
No gem5 PV implementation or new simulated FA result.270 analytical cases:
16/32/64/128/256 nativeFP16 FMA lanes,1/4/16 rows sharingV,1/4/16 prefetched keys,
100/50/25% of existing64B/cycle bank port,35cycleL3/9cycle hypotheticalL2 delay.
Sequential key arithmetic preserved; FMA II1/latency4 is uncalibrated assumption.
FullP-group buffer/PV/O registers explicit;16rows/16keys requires16KiB NEW engine
register state perbank,128KiB across8banks, not added SRAM or free cache storage.
ExistingSPM supplies persistentO/P/V. DestinationP/V writes charged; source/
network/translation/claim/CPU/command/drain timing explicitly NOT predicted.

T4096 busiestbank64000updates independently recovered from accepted stats;
private9473.3184us/currentdirect9037.8088us. Screening target chosen10%lower
latency thanprivate(8525.9866us), not measured or user-prescribed threshold.
16rowreuse/16keyprefetch/50%bankshare synthetic engine+destination-staging:
64lanes10.562ms,128lanes6.466ms,256lanes4.418ms. At128lanes,1/4/16keyprefetch
26.356/8.356/6.466ms. Single-row recurrence dependencies prevent64->256lane
speedup in modeled schedule. Negative residualbudget is a rejection of THIS
schedule, not all possible architectures; mandatory compute/port bounds recorded
separately. Positive budget is not fullFA evidence. Potential candidate128/256
lanes must first substantiate shared-port/register/broadcast/clock feasibility,
then full arithmetic/ordering/resource gates and completeROI vs tunedprivate,
cache/currentdirect, including staging/drain and engine-disabled placementcontrol.
L2-delay-only sensitivity does not establish physicalplacementwinner.

Four scheduling tests PASS compute/port bounds, finiteoccupancy, recurrence
order/width sensitivity, latency/prefetch/bandwidth sensitivity and storagebytes.
Script scripts/diagnose_l3_pv_feasibility.py; sourcehashes/candidatesCSV/JSON/report
results/cacheflex_l3_spm/pv_feasibility_20260930. Single short process,512MiBAS,
CPU44withinexisting24CPUset; resourceguard reserved+512MiB<20%, CPU+onecore<20%,
projectedserverheadroom>64GiB. No simulation restarted/launched, existingqueues
andfrozendependenciesunchanged. Does not by itself settle end-to-end design choice.

### OPS-23 — Shared admission for long context and Qwen remainder (2026-09-30)

**P; user authorized more parallelism under unchanged20%CPU/RAM.** Related
OPS-22 reserved24/48GiB percontext despite~3%actualRAM, delayingAI-12 untilall
recoverytargetscomplete. AI-12 alreadypassed59cases; FA-30existingllama.cpp
reference is a differentprecision/library workload, not a new Llama-model sweep.
This continuation changes host scheduling only, not an architectural experiment.
New shared flock-protected reservationbroker jointly accounts recoveredlimits,
AI-12leases(8GiB each, max4),2GiB AI-12coordinator, oldworkers, llama/outside
margin. Allactors share24CPU affinity. All-userRSS/CPU and >=64GiBserverheadroom
beyondremaining20%userallowance retained. ActiveT16K unusedAS reservations trim
to at least16GiB and >=8GiB above currentVmSize; queuedT32K starts12GiB, expandable
8GiB at4GiBmargin. AI-12children also grow8GiB orpause nearceiling. Noallocation
failure is treated as a performance result. Leases acquired before reference/
launch and held throughacceptance; oldandnewadmissions serialized underonelock.

At most5longcases admitted whileAI-12runs, reservingroomforremainingpipeline;
all7longcaseseligibleafterAI-12finishes. Futurejobreservationreplacesblanket
completiondependency. ExistingPIDs/states preserved bysupervisorhandoff; no
frozenoldcoordinatorresumed. AI-12experiment_memory_v2 reuses59acceptedcases,
sameoriginal run/input/command/describe/cases/unchanged ASTs; oldpublication
block exact.4tests coverconcurrentadmissioncap,lease release,staleheartbeat,
pressure/growth accounting and initialbudget. Allfrozenmanifestschecked before
launch. FirstfullROI independentlyselectsAI-12candidates thenwinner+control
transfers; no new speedupuntil originalfull-output/config/resourcegates pass.
Root results/parallel_overlap_20260930; scientific statuses remainoriginalpaths.

**OPS-23 launch verified:** finalsupervisor4168836, AI-12coordinator4160777.
All9originalsimulatorPIDs preserved; T32Kcache1620709/private1632729 resumed,
NoC/direct resourcequeued. AI-12 firstnewRMSNormSPMb1 accepted,60total; b8/b32
4168899/4168929 running.9simulatorsrunning +2paused. Actual7.08%CPU/3.20%RAM,
145.5GiBfullreservations<150.987GiBcap,428GiBadditionalserverheadroom. No new
failures. Alloriginalmanifestfileschecked;4broker tests passed; exactoriginal
publication/acceptance routines retained. Oldsupervisors3771764/4160769 stopped,
neverresume. Newjobs fill brokerleases asquota becomesavailable; max4remainder.

**FA-30 status correction / OPS-23 reserve release:** older upstreamllama.cpp
T16K run27574 had exited on48h timeout, recorded in originalstatus(updated
1790799162.8628). No accepted T16K upstreamtiming; failedpartial preserved.
Verifiedno live associatedsimulator/recoverycoordinator; released stale9.5GiB
reservation throughsame-mainloop v2handoff. This is not a Llama-model benchmark
completion or grounds for pooling its differentarithmetic withQwen. First
v2preparationguard matched itsown shell text; corrected toactualprocessargv,
no simulationsaltered; preparation-onlymissingfile launchlogretained.

**OPS-21 ffn complete:** 28 accepted cases; original acceptance unchanged. Effects [{"t": 1024, "cache": "screen_t1024_cache_t1024_mp8_kc512_g0", "spm": "screen_t1024_spm_t1024_mp8_kc1024_g0", "cache_us": 26616.9476, "spm_us": 19592.028, "speedup": 1.3585601041403168}, {"t": 2048, "cache": "target_t2048_cache_t2048_mp8_kc512_g0", "spm": "target_t2048_spm_t2048_mp8_kc1024_g0", "cache_us": 50082.744, "spm_us": 45632.962, "speedup": 1.0975124516352894}, {"t": 4096, "cache": "target_t4096_cache_t4096_mp8_kc512_g0", "spm": "target_t4096_spm_t4096_mp8_kc1024_g0", "cache_us": 118674.6728, "spm_us": 91951.0536, "speedup": 1.2906287438124582}].

### FA-33 — Combined private Q/output residency and balanced per-head Qwen FA (2026-09-30)

**P; user authorized implementation.** Related FA-10, FA-25–29, FA-14, L3-06/07/08/11/12. New combined Q+O (192KiB/core at4096), with/without V (224KiB), complementary query-block ownership, one head at a time. Same-layout coherent/V controls and same-ELF legacy controls; frozen hardware unchanged. Full independent/exact-output/config/count/finite-resource gates before fullROI comparison. No accepted result. [Plan](QWEN_QO_RING_PLAN.md).

### FA-34 — Neighbor-SPM ring delivery feasibility (2026-09-30)

**P; authorized downstream of FA-33.** GEMM-05 multicast already reduced copies with modest gain; FA-14 buffering regressed. New neighbor-to-neighbor SPM transport plus resident Q/O. Optimistic headroom, bounded correctness, realistic port/link/credit model, conservative sensitivities then second workload. Decision is lower complete ROI vs same-layout ordinary staging and strongest private/cache, not traffic alone. No hardware or ring result yet. [Plan](QWEN_QO_RING_PLAN.md).

**FA-33 implementation update:** guest_v2 compiled;7 host guardsPASS.
experiment_v3 coordinator554757 repeats all6T256 gates thenT1024/T4096 factorials,
max2additional simulators under OPS-23 atomic reservation/20%CPU-RAM constraints.
V2 smallcache accepted126.414us; QO allzero failed because private SPMST does not
implicitly claim slots. Guest fixed to stage zero O via4real64Bcopies/row plus
normal drain, with counts/timing included; Q2copies/row. Failed evidence retained,
no hardware or numerical tolerance change. No accepted QO performance yet.
FA-34 remains an authorized dependent stage, not an implemented ring model.

**AI-12 memory-recovery continuation complete:** [{"op": "rmsnorm", "t": 1024, "cache": "screen_rmsnorm_cache_t1024_v151936_b32", "spm": "screen_rmsnorm_spm_t1024_v151936_b32", "cache_us": 52.9512, "spm_us": 52.9512, "speedup": 1.0}, {"op": "rmsnorm", "t": 2048, "cache": "target_t2048_rmsnorm_cache_t2048_v151936_b32", "spm": "target_t2048_rmsnorm_spm_t2048_v151936_b32", "cache_us": 305.4544, "spm_us": 305.4544, "speedup": 1.0}, {"op": "rmsnorm", "t": 4096, "cache": "target_t4096_rmsnorm_cache_t4096_v151936_b1", "spm": "target_t4096_rmsnorm_spm_t4096_v151936_b1", "cache_us": 730.8084, "spm_us": 730.8084, "speedup": 1.0}, {"op": "residual", "t": 1024, "cache": "screen_residual_cache_t1024_v151936_b8", "spm": "screen_residual_spm_t1024_v151936_b8", "cache_us": 162.6004, "spm_us": 162.6004, "speedup": 1.0}, {"op": "residual", "t": 2048, "cache": "target_t2048_residual_cache_t2048_v151936_b1", "spm": "target_t2048_residual_spm_t2048_v151936_b1", "cache_us": 494.5644, "spm_us": 494.5644, "speedup": 1.0}, {"op": "residual", "t": 4096, "cache": "target_t4096_residual_cache_t4096_v151936_b1", "spm": "target_t4096_residual_spm_t4096_v151936_b1", "cache_us": 1125.3344, "spm_us": 1125.3344, "speedup": 1.0}, {"op": "rope", "t": 1024, "cache": "screen_rope_cache_t1024_v151936_b32", "spm": "screen_rope_spm_t1024_v151936_b32", "cache_us": 93.2836, "spm_us": 93.2836, "speedup": 1.0}, {"op": "rope", "t": 2048, "cache": "target_t2048_rope_cache_t2048_v151936_b32", "spm": "target_t2048_rope_spm_t2048_v151936_b32", "cache_us": 360.2044, "spm_us": 360.2044, "speedup": 1.0}, {"op": "rope", "t": 4096, "cache": "target_t4096_rope_cache_t4096_v151936_b32", "spm": "target_t4096_rope_spm_t4096_v151936_b32", "cache_us": 934.3024, "spm_us": 934.3024, "speedup": 1.0}, {"op": "rope_tables", "t": 1024, "cache": "screen_rope_tables_cache_t1024_v151936_b32", "spm": "screen_rope_tables_spm_t1024_v151936_b32", "cache_us": 40.0096, "spm_us": 40.0096, "speedup": 1.0}, {"op": "rope_tables", "t": 2048, "cache": "target_t2048_rope_tables_cache_t2048_v151936_b32", "spm": "target_t2048_rope_tables_spm_t2048_v151936_b32", "cache_us": 79.9632, "spm_us": 79.9632, "speedup": 1.0}, {"op": "rope_tables", "t": 4096, "cache": "target_t4096_rope_tables_cache_t4096_v151936_b32", "spm": "target_t4096_rope_tables_spm_t4096_v151936_b32", "cache_us": 159.1572, "spm_us": 159.1572, "speedup": 1.0}, {"op": "embedding", "t": 1024, "cache": "screen_embedding_cache_t1024_v151936_b1", "spm": "screen_embedding_spm_t1024_v151936_b1", "cache_us": 115.4464, "spm_us": 115.4464, "speedup": 1.0}, {"op": "embedding", "t": 2048, "cache": "target_t2048_embedding_cache_t2048_v151936_b1", "spm": "target_t2048_embedding_spm_t2048_v151936_b1", "cache_us": 328.8724, "spm_us": 328.8724, "speedup": 1.0}, {"op": "embedding", "t": 4096, "cache": "target_t4096_embedding_cache_t4096_v151936_b1", "spm": "target_t4096_embedding_spm_t4096_v151936_b1", "cache_us": 750.6944, "spm_us": 750.6944, "speedup": 1.0}, {"op": "final_norm", "t": 1024, "cache": "screen_final_norm_cache_t1024_v151936_b32", "spm": "screen_final_norm_spm_t1024_v151936_b32", "cache_us": 52.9512, "spm_us": 52.9512, "speedup": 1.0}, {"op": "final_norm", "t": 2048, "cache": "target_t2048_final_norm_cache_t2048_v151936_b1", "spm": "target_t2048_final_norm_spm_t2048_v151936_b1", "cache_us": 304.7064, "spm_us": 304.7064, "speedup": 1.0}, {"op": "final_norm", "t": 4096, "cache": "target_t4096_final_norm_cache_t4096_v151936_b1", "spm": "target_t4096_final_norm_spm_t4096_v151936_b1", "cache_us": 731.6424, "spm_us": 731.6424, "speedup": 1.0}, {"op": "logits", "t": 1, "cache": "screen_logits_cache_t1_v151936_b1", "spm": "screen_logits_spm_t1_v151936_b1", "cache_us": 11844.1604, "spm_us": 11844.1604, "speedup": 1.0}]
Full-output/config/resource gates; shared coherent SVE implementation, no L3 offload. Integration remains pending.

**FA-33 experiment_v3 accepted-only outcome:** stage=completed_validated, accepted=18. All accepted subsets retained; failed/cancelled cases excluded. T1024 mode0 859.3648us (0.952653x priorV); T1024 mode1 832.1000us (0.983868x priorV); T1024 mode2 846.1680us (0.967511x priorV); T1024 mode3 844.2808us (0.969674x priorV); T1024 mode4 1249.7248us (0.655086x priorV); T1024 mode5 1131.4500us (0.723564x priorV); T4096 mode0 10437.5852us (0.907616x priorV); T4096 mode1 9644.1068us (0.982291x priorV); T4096 mode2 10494.6368us (0.902682x priorV); T4096 mode3 9523.5580us (0.994725x priorV); T4096 mode4 14321.4552us (0.661477x priorV); T4096 mode5 10306.1280us (0.919193x priorV)
[Effects](../../results/cacheflex_flash_attention/qwen_qo_ring_20260930/experiment_v3/EFFECTS.md). No ring performance claim.

### MOD-12 — Placement/ownership analytical model and failed timing validation (2026-09-30)

**D/N accounting; timing predictor fails validation.** User authorized numerical
placement/ownership selection. Related FA-03/10/14/25–29/31/33 and L3-11/12:
those already tested displacement, Q adapters, output residency, asynchronous
staging, KV partitioning, finite-profile tuning and the current Qwen factorial.
New model enumerates ownership, residency lifetime, Q/V/O/KV placement, tile
sizes and physical slot mapping together. Exact per-core tasks/updates/copies
match all18 accepted FA-33 cases. Static K+V atT4096 needs128KiB/core perKVgroup
or256KiB forbothheads; BOTH copy2MiB total K/V payload once. No raw-sum merge:
software merge requires FP32 partial output plus max/denominator rescaling.

SixT1024 cases calibrate positive global scale over explicit resource scenarios;
sixT4096 cases withheld from fitting. Initial model MAPE22.4%, pairwise ordering
21.4% (ignoring measured pairs within1%); fails expected placement directions.
Full phase graph is an aggregate approximation, not a finite-queue simulator;
cache/set-conflict and KV-merge timing uncalibrated. This is not a blind external
study: T4096 behavior was known during design. No refit to disguise failure.
Runtime ranks disabled for performance selection; scenario envelopes are not
confidence intervals or guaranteed bounds. Failed validation retained.

Information-gain plan: QV,OV,KV-owned cache,staticKVgroup,staticbothheads. New
guest/runner implementations explicitly missing; no simulations launched or
existing queues/frozen dependencies modified. A candidate must lower complete
ROI versus matched control AND strongest priorprivate9473.3184us before it
changes the performance decision. Full numerical/config/count/resource gates
and shared20%CPU/RAM broker required for future execution.

[Model report](../../results/qwen_placement_model_20260930/REPORT.txt),
[plan](../../results/qwen_placement_model_20260930/experiment_plan.json),
[design/usage](QWEN_PLACEMENT_MODEL.txt). Source snapshots/provenance retained.

### FA-35 — Bounded T4096 Qwen placement and geometry matrix (2026-09-30)

**P; implemented, runtime validation queued.** User authorized missing placements
and justified Br/Bc/R search. FA-33 already found QO/QOV slower than V; FA-10/25–29
cover prior Q/output adapters and KV partitioning; L3-11/12 cover Qwen Br16
neighbors. MOD-12 exact accounting is usable but its timing ranking failed.
New QV/OV and static K+V (one KV group or both heads), matched KV-owned coherent
control, variable balanced-head R, Br16/32/64, Bc128/256/512 and R grids. Existing
cache/V/QO/QOV/L3 cases are revalidations with one new ELF, not new mechanisms.
12 anchors +72 axial cases +up to4 measured interactions and2 neighbors/family:
maximum156 T4096 targets, separate deduplicated full-output gates. KV Bc512
rejected by ownership geometry. No model-based runtime pruning or global optimum
claim. All outputs, exact same-arithmetic/Bc bits, config/count/finite resources
required; failed cases stop downstream. Static KV has independent softmax-merge
oracle. Decision: lower complete ROI versus matched controls and strongest prior
private9473.3184us/cache10509.4604us. No ring or hardware change.
11 host tests pass;387 frozen inputs verified. Shared OPS-23 broker, max4 new
simulators within20% CPU/RAM and64GiB server headroom. Existing runs untouched.
[Design](QWEN_FA_MATRIX.txt); results/cacheflex_flash_attention/qwen_matrix_20260930/
experiment_v1/status.json is authoritative; no accepted runtime result yet.

**FA-35 launch verified:** coordinator1312454; QV1312483/OV1312514/cache1312557/V1312575 alive, unpaused. Initial full-output gates pending; no new performance claim. All-user CPU7.01% ofhost, RSS37.685GB (~4.65%RAM), pressurefalse; combined reservations148.713GB below162.121GB cap.

**FA-35 v1 stopped; v2 continuation (2026-09-30):** all9 query-owned
T256 gates accepted full checks. Staticboth KV failed before acceptance on
SPM read0x60050000/256B crossing way8. KVgroup/cache peers cancelled; no KV
performance accepted. At final Vset1023, linear next iteration advances way4
into way5; add explicit set-wrap BFI to static-V PV loop, preserving every
FMA and legal in-loop address. Bounds test includes speculative lookahead;
full root-cause trace not claimed. New guest_v2; oldguest/source/results kept.
12host tests PASS. New run_qwen_fa_matrix_v2.py repeats gates/targets with same
new ELF. User requested more parallelism: cap5 via unchanged atomic20%broker,
24CPUaffinity,64GiB extra headroom. Five8GiB leases plus existing114.354GB reserve
fit162.121GB cap; six do not. Fix cleanup to release all failed-peer leases;
removed one stale v1 lease only after no live v1 simulator/coordinator verified.
No simulator/model/timing/queue change. No new target performance claim.

**FA-35 v2 gate acceptance:** all12 anchors' runtime gates PASS full independent
output/config/count/finite-resource checks, including KVcache/group/both at
T4096/Hq2/Hkv2. Corrected static-V address path accepted. Query gates useT256;
KVgate timings2809.6432/2874.7276/3477.4024us are NOT full14headtarget results.
Five fullT4096/Hq14 anchors active; seven remaining anchors queued, axial then
adaptive phases follow. No full-model-size matrix performance acceptance yet.

**L3 long-context T16384 direct accepted:** 133423.6508us versus
cache154969.2524/private142105.2900/NoC133777.5472us. All14680064outputelements
checked, zeroerrors and exact output hash across arms, original count/config/
resource gates retained. Direct saves6.109%latency vsprivate and0.2645%vsNoC;
1.16148xcache. Transfer Br16/Bc256/R256, not independently retuned16K optimum.
T32768fourarms remain active; no accepted32Ktiming yet.

**FA-35 scheduling-only v3 handoff:** user requests operation closer to20% with
server headroom. Accepted16Kdirect released old reservations81GiB (was106.5).
Eight matrix leases now fit with four32Ksimulators. Newcoordinator1485188 adopts
five existing T4096children, unchangedPIDs/inputs/commands, and admits pending
anchors. Oldcoordinator1397296 SIGSTOPPED, do not resume; retired only after
adoptedchildren exit status and normalfull acceptance captured. No guest or
model modification.12kernelhosttests and adoptedexitstatus gatePASS. Shared
broker cap150.987GiB,24CPUaffinity and64GiBextra headroom remain; growth reservations
retained. experiment_v3/status.json authoritative, oldv2statusstale. Materialdelta
is host scheduling ONLY; end-to-end ROI acceptance/ranking unchanged.

**FA-35 v3 admission verified:** five adoptedchildren continue unchanged; two additional anchors launched (headV1485926/headQO1485964), sevenmatrix +four32K=11live. An intervening memory-growth reservation leaves145GiB/150.987GiB reserved, so eighthmatrix admission correctly waits. Actual all-userCPU8.66%,RSS38.125GB(~4.70%), pressurefalse. Cap8 is a ceiling, not unconditional admission.

**FA-35 v3 stopped (2026-10-01):** 12initialgates+11fullT4096/Hq14anchors accepted. QO target fails SPM read0x60050000/256B crossingway8; no acceptedQOtiming, axial/interactions/neighbors NOT started. This is a separate Q path from the earlier repaired static-V path; exact originating instruction/trace pending. Accepted subsets retained; no global baseline freeze yet. FullROIus: t4096_q14_legacy_v_br16_bc256_r256=9418.3056, t4096_q14_head_qv_br16_bc256_r512=9921.3184, t4096_q14_legacy_cache_br16_bc256_r256=10432.5348, t4096_q14_head_cache_br16_bc256_r512=10467.7876, t4096_q14_head_v_br16_bc256_r512=9674.652, t4096_q14_legacy_l3_br16_bc256_r256=9391.4172, t4096_q14_kv_group_br16_bc256_r64=19975.6712, t4096_q14_kv_cache_br16_bc256_r64=18932.3156, t4096_q14_head_ov_br16_bc256_r512=10065.528, t4096_q14_head_qov_br16_bc256_r512=11921.7712, t4096_q14_kv_both_br16_bc256_r64=21404.0752. LegacyV 1.107687xmatchedcache; currentL3anchor0.28549%lowerlatency vsV, not best-tunedL3conclusion. StaticKVgroup/both lose KVcoherent and V at these anchors. All four32K arms stillactive; no32K acceptance.

**FA-35 Q-boundary repair (2026-10-01):** separate from guest_v2 static-V fix.
Q adapter previously AND0x7ffc0 retained waybits: last staged Qrow ends0x4ffff,
next loop lookahead0x50000 selectsway5; physical256Bread spansways5..8 andpanics.
New guest_v3 masks set/line bits0xffc0 then OR0x40000, pinning all6Qloadsites
in way4 (physicalreadways4..7). All consumed addresses and FMA instructions
unchanged; extra address instruction cost INCLUDED in new-ELF comparisons.
14hosttests PASS incl all valid Qaddresses/overflow mapping and exact adapter
diff;387frozeninputsverified. Build/guest/hardwareoldfiles untouched. Dynamic
trace proving exact speculative instruction not claimed; source path identified.
experiment_v4 first runsT4096/Hq2/R512 QO/QV/QOV full-capacity gates, then repeats
same-new-ELF controls/anchors withQO first, thenaxial/interactions/neighbors.
Same fulloutput/exactbits/config/count/resource acceptance. Max8matrixjobs only
when shared20%broker admits; four32K arms untouched. Decision: failedQOcase must
complete fullvalidation, then completeROI determinesbaseline; no new timing yet.

### VER-01 — Current Rumur refresh and bounded rerun (2026-10-01)

User authorized updates/rerun after CV scope audit. Related MOD-08 correctness
repairs, FA-31 retry/placement, L3-02/03 and lifecycle ordering fixes; prior
verification/rumur/cacheflex_spm_mc.m remains historical. No hardware or guest
change. New current/private_spm.m includes local-source PUT/ack relinquishment
(the old model incorrectly retained it), silent NACK/existing-TBE retry, coherent
writeback invalidate/ack, retained LLC directory ownership, transient migration,
and active coherence checks without sticky wb_race exemption. Silent registration
check is attributed to the request transition, not a later completion at which
a prior ordinary Unblock can legitimately change metadata.

Focused network model is ONLY a nine-state directed dependency witness, not a
Garnet/full-network proof. Focused L3 model checks symbolic command/slot ordering,
not FP arithmetic or complete CPU/LSQ/protocol equivalence. Negative variants
remove each safeguard and must produce a named trace; crashes/OOM not passes.

Runner scripts/run_rumur_refresh_v5.py, root results/rumur_refresh_20261001,
verification_v5 authoritative. Two checker threads, one case at a time, shared
20%broker reservations and AS limits; larger bounds wait. Fourteen cases include
2core2/3/4/5operations,3core3operations, retentionoff, network/L3 positives and
negativecontrols. Frozen source/tool/model hashes, generated C, logs, exit status
and state counts retained. Historical evidence is not overwritten. v1 superseded
on source-semantic audit; v2 exposed missing modeled directory ownership and
PUT-wait permission classification; v3 type-name generation failure; v4 exposed
silent-registration ghost attribution. These do NOT count as accepted current
verification. v5 runtime underway; no all-bounds pass claim. Existing FA/32Kruns
untouched. Root run_cacheflex_spm.sh no longer defaults to all host threads.

Decision criterion: named invariants/liveness pass within documented bound and
negative controls detect removed guards. Any unexpected counterexample requires
classification/model-to-source tracing; do not call it a silicon bug or weaken
checks merely to pass. [Scope](../../verification/rumur/current/README.md).

**VER-01 v5 accepted subset:** 11/14 cases accepted; zero unexpected findings.
Private2core2op43071states and2core3op718595states PASS; retentionoff2core3op
703141states PASS. Disabling WB invalidation yields expected freshness/SWMR
counterexample90164states. L3positive625states; all3removed-ordering controls
produce namedcounterexamples. Directed networkpositive9states; disabling retry
or VC-class separation produces expectedliveness witnesses. These tiny focused
models are explicitly not completenetwork/LSQproofs. Remaining3core3op,2core4op,
2core5op automatically wait for4/4/12GiB reservations. Report via
scripts/report_rumur_refresh.py. Lightweight brokercontinuation watcher remains
read-only until original supervisor exits cleanly; then maintains existing
ledger/limits if FA/formal leases remain, preventing stale-heartbeat queue stall.
No reservation removed or expanded beyondcap; no simulator edited/restarted.


### Repository cleanup — operational, no new experiment (2026-10-01)

User requested CV removal, obsolete artifact cleanup, and benchmark organization.
Related FA-35 and VER-01 inputs remain immutable. All 493 distinct inputs across
current FA-matrix, long-context and Rumur manifests matched before and after.
Removed 48 CV/empty artifacts; archived and extraction-hash-verified 428 obsolete
root-build/Rumur generated products, root build logs, and superseded Rumur v1-v4
launchers. Net file-byte reduction 149464576 (~142.5 MiB), retaining a 20542130-byte
recovery archive. Source models, current gem5 incremental build, accepted/failed
simulation evidence and active processes are preserved. New benchmarks/qwen,
verification and infrastructure navigation links avoid path changes; root and
benchmark READMEs now identify current entry points. Audit/recovery:
archive/workspace_cleanup_20261001/complete.json and manifests. This changes no
ROI, hardware model, guest, acceptance gate or resource scheduling policy.


**FA-34 implementation / queued headroom diagnostic (2026-10-01):** FA-33
prerequisite residency factorial completed, but had not implemented ring transport.
New benchmarks/qwen/fa/ring adds actual K/V neighbor data rotation among already
claimed private SPM slots, explicitly ZERO modeled transfer service/port/link
latency; timed CPU arithmetic, initial coherent copies, DSB/barriers, packing,
normalization and release remain. No added tile SRAM. Q64KiB/O128KiB/K32KiB/V32KiB
at T4096 fits existing private slots. Cyclic eight-panel waves, seven hops each,
complementary per-core query ownership; future query work skipped but forwarding
participation retained. This is schedule-specific optimistic headroom, not a
realistic hardware result or a universal upper bound.

Two matched ordinary-staging/ring pairs: Q/O resident and Q/O coherent, all with
same cyclic visitation and K/V SPM layout. Same-new-ELF legacy cache/V controls
included. Independent rounded-FMA oracle follows cyclic order, FP64 dense check
at T1024, matched cyclic output bits, exact copy/rotation/byte/work counts and
finite-resource/config gates required. T1024/Hq2 gates then full14-head T1024 and
T4096. Two old-ELF simulator exact controls and directed normal/delayed-receiver/
missing-claim-negative tests precede all FA. Decision: complete ROI versus matched
ordinary delivery AND strongest accepted V-only, not isolated transfer savings.

Guest_v1 and directed ELF compiled; six host checks pass. Isolated pseudo_inst
replacement object build_v2 compiled without changing main gem5 source, objects
or active simulators. V1 superseded BEFORE linking/simulation: source audit
corrected raw ISA slot addresses to the existing 0x60000000 Ruby SPM namespace.
No runtime ring result accepted yet. Coordinator57426 experiment_v2 waiting for
4GiB link reservation; then serial gated23case queue under same20%broker, initial
8GiB simulation lease, two host CPUs/one simulator, admission-controlled growth.
Root results/cacheflex_flash_attention/optimistic_ring_20261001. Status.json is
authoritative. Old FA-33, FA-35, long-context, Rumur evidence/queues unchanged.


**Host execution reorganization and FA-34 build retry (2026-10-01):**
Host scheduling only; no new architectural experiment or accepted ROI result.
Related queues FA-34/FA-35 and VER-01 retain their frozen inputs and validation.
New supervisor762804 adopts all existing simulator PIDs; old4168836 remains
intentionally stopped. Reclaimed12GiB of unused reservation allowance, reduced
base reserve11->6GiB after checking outside RSS, and changed long-context growth
to1GiB increments with2GiB proactive margin. Adopted AS limits retain>=3GiB slack.
Same total reservation/actual-RSS/CPU/headroom guards. All six existing simulations
and the next Rumur checker now run; measured host usage about5.6%CPU/13.1%RAM.
Result acceptance/publication functions unchanged (AST audit); source manifests
verified. Evidence: results/resource_scheduler_20261001.
FA-34 build_v2 then failed linking at3GiB process AS (gold out of memory), before
any simulation. This is failed host-build evidence, not a performance result.
Same compiled object/guest queued as build_v3/experiment_v3 via run_v3.py,
coordinator771311,8GiB reservation/7GiB linker AS, single linker thread and reduced
file mapping. All23 runtime cases remain pending. No numeric gates relaxed.

**VER-01 unexpected liveness finding traced (2026-10-01):** v5 completed with
7 positive passes,6 expected negative-control failures,1 unexpected failure
(private_n2_s5;45,604,056states/236,535,184rules). Two source-release PUTs remove
all LLC sharers but leave SS. Subsequent SPMWB enters SPM_WB_INV with zero pending
acks/no invalidation recipients. Core0 can consume a remaining Put_Ack, but Core1
cannot finish WB. Direct source correspondence: LLC SS PUT transitions1707/1714,
sharer removal1444, WB TBE count1272, unconditional SS/MT WB transition1893,
response-only completion1904. No automatic zero-count completion in setState or
response dispatcher. Model/source hashes all unchanged. Source-level defect,
not yet directed gem5 reproduction; do not claim model-only artifact or full
hardware reproduction. Proposed narrow fix is empty-sharer WB direct path,
retaining invalidation for live copies and directory acknowledgment. No fix or
new simulation launched in this diagnosis. Evidence and reconstructed states:
results/rumur_refresh_20261001/verification_v5/diagnosis_zero_sharer/REPORT.txt.


### VER-02 — Retained-source SPMCP (2026-10-01)

User authorized copy-without-source-release semantics after VER-01 zero-sharer
counterexample. Current verification_v6 runner3798468:11/15 accepted so far,
no unexpected findings;3core3op running,2core4op/5op and old-release5op control
queued. Old v5 failure preserved. New invariants: no source PUT under retention,
pending snapshot belongs to outstanding copy. Do not edit frozen sources while
this queue runs. Source/model snapshots at results/spm_copy_retention_20261001.
L0 hit now carries64B snapshot in existing CoherenceMsg (Data size + snapshot flag),
keeps coherent copy; privateL2 hit buffers in existing TBE before lazy migration,
retains state/registration. Absent source GETS_SILENT unchanged; old L0-eviction
recall fallback retains private ownership. SPMWB unchanged. No added queue/SRAM.
SLICC generation and6generated controller C++ syntax checks PASS. Full gem5 binary
build/directed runtime validation NOT done; no performance result or baseline
replacement claim. Existing running simulations/binaries untouched. Max2checker
threads, shared20%broker and4/12GiB large-case reservations. Current status:
results/rumur_refresh_20261001/verification_v6/status.json; old v5 historical.

Related VER-01 and MOD-08: former source-release model passed smaller bounds
but failed the2core5op drain property. This is a protocol-semantic change, not a
performance tuning point. It tests whether retaining coherent source state and
registration removes that failure while preserving snapshot freshness, SWMR,
writeback invalidation and liveness. The15th case restores release semantics
(RETAIN_SOURCE=0), expected to reproduce the old named liveness failure. Full
same-model positive/negative acceptance is the decision criterion. No ROI claim;
future performance comparisons need matched cache/SPM revalidation on a built
and runtime-validated implementation. Original simulations are not retroactively
relabelled as measurements of this change.


**VER-02 formal completion and protocol revalidation restart (2026-10-02):**
All15v6 cases accepted:8positive passes and7expected negative counterexamples.
Retained2core5op passes48,275,746states/250,746,538rules; RETAIN_SOURCE=0 restores
old liveness failure at45,604,056states/236,535,184rules. This supports bounded
retention semantics, not full simulator/protocol refinement or hardware proof.
User explicitly cancelled old32K and requested restart under changed protocol.
Related FA-35/L3-10 retain108/8 historicalaccepted cases respectively. Oldprivate
32K exited0 but validator host1.5GiB AS failed; remains unaccepted, preserved.
All oldactive simulations cancelled; no invalid old timing transferred as new.
New root results/retained_protocol_restart_20261002: isolatedbuild then13directed
gates then freshfullFA matrix and four-arm T1024/T16K/T32K long-context runs.
Matched binary/guests/configs/full arithmetic/exact sameBc output/counts/finite
bounds retained. EightGiB validator budget fixes hostmemory headroom without
changing oracle code. Max2FA+1long, shared20%CPU/RAM/64GiBheadroom. Expected
protocol latency/count changes mean oldall-stats exact equivalence is not an
acceptance criterion for the new semantics; new fullROI measurements needed.
No new runtime gate or performance result accepted yet.


**VER-02 runtime gates / FA-35 restart (2026-10-02):**
Current coordinator1661754, matrix1662752; source/model unchanged sincev6.
New binary SHA98381f98474293706a03725efe06c7876f611e3785b67299afad7c1309f7dca9.
All13 runtime gates PASS: privatecopy/WB/release/raw/placement, sharednormal/small
queues, directreductionFP32/both/init/mixed/oneentry/sync. Roundtrip exercised64
M.SPMCP_snapshot transitions. Buildfix explicitly runs intercepted link manifest;
initialcopiedoldELF NEVER simulated. Preservedpackaging/harnessfailures underroot.
Oldgateconfigs upgraded to complete wrapper; all existing configfields identical,
new disabledenginequeues explicitlybounded4. Larger-build optionalparameters
recorded, no performance claim from directedgates. Sourceinputhashes verified.
FAmatrix now running firstT4096Q2 QO/QV gates at1675575/1679095, then remaining
QOV gate and full14head bounded anchors/axial/interactions/neighbors. Sameguest,
fulloracle/config/copy/count/bounds checks, newtimings only. max2FA concurrent.
Newlong-context runner alsoactive, revalidate_cache first (thenother3smallcontrols,
16K/32K). Fixed launch-onlysmallcontrol geometry guard toapplyT>1024; no numerical
checks relaxed. Originalfailedlaunches had no acceptedperformance. Status root:
results/retained_protocol_restart_20261002/{status,build_status}.json and
fa_matrix/status.json,long_context/status.json. Oldqueues remaincancelled.



**VER-02 / FA-35 / L3-10 audit repair (2026-10-02):**
Coordinator1661754 stopped peers after completed T1024 cache failed host config
validation, not simulation. Preserved long_context_config_failure and
fa_matrix_config_failure; zero performance records accepted in those attempts.
Verified exactly17 config differences:16 added bank read/reply overrides=0
(source selects original l3_spm_latency), plus expected frozen guest executable.
Guest driver differs only T upper bound16384->32768; all shared headers identical.
Audit now permits ONLY missing-old-field additions at zero; existing changes still
fail. Three negative checks reject nonzero override, changed capacity, wrongguest.
Evidence audit_failure_preserved/verified_differences.json. All225 prior input
hashes verified before modification; original runner saved. Completed cache output
is copied and fully revalidated (command/input hashes, full oracle/counts/bounds);
no simulation rerun or acceptance shortcut. Partial matrix gates restarted.
Coordinator2461257, matrix2462111,long2462112; same simulator/guests/resources,
max2FA+1long,20%CPU/RAM and64GiBspare. Current status files authoritative.

Saved T1024 cache successfully accepted after full oracle/export/config/count/bounds checks; private control2468917 and matrix QO2473726/QV2476997 now running. No active pauses; user RSS5.35GiB, below151GiB cap. This cache revalidation alone is not a new speedup comparison.


**VER-02/FA-35/L3-10 matrix audit repair:** The independent matrix validator also rejected the16 added zero read/reply overrides. Fixed only missing-old-field zero additions, with exact guest/argv checks. Saved matrix config passes; four negative checks reject nonzero timing, changed capacity, wrong guest and wrong argv. Both prior manifests (580matrix/561long entries) verified before preserving attempts at fa_matrix_matrix_audit_failure and long_context_matrix_audit_failure. Frozen replay dependencies in matrix_audit_repair/verified_replay_inputs.json. Replay completed QV gate and accepted cache/private outputs through full validation; restart partial QO and NoC. New coordinator1438251/matrix1441430/long1441431; unchanged simulator/guest/caps. Prior failures preserved, not performance evidence.
Same-guest historical exact_cache/exact_private controls at qwen_long_context_20260928/experiment_v1 are848.8716/818.6768us versus new848.8716/819.4732us. Cache identical, private0.0973% slower; speedup1.036882->1.035875x. Small T1024 result only; no accepted full14head4K result yet. Evidence matrix_audit_repair/protocol_comparison.json.

**SPM-only continuation (user correction, 2026-10-02):** User clarified only SPM runs, not cache. Mixed coordinators stopped and results preserved. New coordinator2387013, matrix2389368,long2389369; authoritative spm_only_status.json, fa_matrix_spm_only/status.json and long_context_spm_only/status.json under retained_protocol_restart_20261002. Cache modes0/2/9 filtered from every phase and selection; private/shared/reduction modes retained. Local selection helper restricts families to avoid empty cache-family winners. Long queue excludes all cache and already accepted private/NoC T1024; resumes direct then SPM/NoC/direct16K/32K. One accepted QV gate and two accepted smallSPM controls retained after frozen-input verification, no reruns. First SPM-only launch stopped before acceptance to restrict selector, evidence *_launch1. All frozen model/oracle/resource constraints retained: max2matrix+1long,20%CPU/RAM,64GiBhostheadroom. No new speedup claim; related VER-02/FA-35/L3-10.

**FA-34 retained-protocol rebuild (2026-10-02):** User authorized fixing ring build/queue. Old v3 link failed at7GiBAS before any simulation. New build_retained_v1 overlay recompiled against isolated retained_protocol_restart_20261002/gem5 sources/generated headers; same optimistic hook/guest, main sources/objects and active queues unchanged. Six host tests PASS, overlay compile PASS with2GiBreservation. New run_retained.py coordinator505214, experiment_retained_v1:24GiB link lease/23GiBAS, one linker thread, reduced mapping; later8GiB simulation lease/6GiB initial AS and admitted growth. Shared20%CPU/RAM,64GiBspare, two allowed CPUs/one simulator.19cases, SPM-only per user: retained-profile exact private control, normal/delayedreceiver/missingclaimnegative directed gates, then mode1/12/13/14/15 at T1024Q2,T1024Q14,T4096Q14. Mode0cache excluded, no old-protocol exact-stats requirement. Exact private now references accepted updated-protocol T256V control, frozen before linking. Full independent cyclic/dense math, matched-outputbits/config/copy/rotation/byte/work/resource checks retained. Decision full ROI versus matched ordinary delivery and V-only; zero modeled transfer service remains a schedule-specific headroom test, not realistic hardware. No runtime ring acceptance yet.

FA-34 retained build LINK PASS, SHA7b2e73d8c02a4cf7fcb5f26339b85d88efe04a6daf8e1e22876c4a60920d453a. First exact_private simulator647177 running, then directed gates and matched FA. No accepted runtime result yet; experiment_retained_v1/status.json authoritative.

**FA-34 retained T1024 full14-head completion:** All five arms accepted full FA checks: V1076.3652us, QO ordinary1636.2848/ring1493.3156us, coherent-QO ordinary1497.8044/ring1344.8956us. Ring saves8.74%/10.21% versus matched cyclic ordinary delivery, but best ring remains24.95% slower than same-ELF V control. Zero-transfer-cost diagnostic, not realistic hardware. All directed gates accepted including intentional missing-claim failure. T4096 five arms now active/queued; no T4096 ring conclusion. FA-35 retained first14head target QV Br16/Bc256/R512 accepted10045.594us; remaining targets pending. Current experiment_retained_v1 and fa_matrix_spm_only statuses authoritative.


### Result-history maintenance / ongoing retained-profile results (2026-10-02)

User explicitly requires checking prior valid evidence before new experiments to
avoid redundant work. This ledger remains authoritative across historical runs.
`CURRENT_RESULTS.txt` and `current_results.json` supplement it with individually
linked validator-accepted records from ongoing FA-35, L3-10 and FA-34 queues.
Refresh with `python3 scripts/refresh_current_result_log.py` on status/result
reviews and before experiment decisions. Initial refresh: 41 accepted records,
zero status/acceptance-marker timing discrepancies. This checks record provenance,
not a second independent validation of arithmetic. No active runner was modified.

**FA-35 accepted subset, tuning pending:** retained-protocol T4096/Hq14/Hkv2
anchors include V-only9418.3056us, legacy-L3 9391.4172us, head-V9667.4292us,
QO15614.7864us, QV10045.5940us, OV9991.8280us, QOV12025.2900us and
static KV-group18851.2148us. Exact geometries/evidence are in current_results.json.
L3 saves only0.28549% against its matched V anchor. These are starting placements,
not final independently tuned winners. Every included SPM family proceeds through
bounded axial/interactions/neighbor tuning; do not repeat existing anchors.

**FA-34 accepted subset:** same ring guest T4096/Hq14 V-only9423.5016us;
Q/O-resident ordinary-delivery13307.8688us versus optimistic-ring12773.3000us
(about4.02% lower full ROI, still slower than V-only). Q/O-coherent ordinary
12391.6680us accepted; corresponding ring case pending at this snapshot.
Zero-time movement is a schedule-specific upper-bound experiment, not measured
physical ring performance. Do not combine its controls with FA-35 as if same ELF.

**L3-10 accepted subset:** retained-protocol T1024/Hq14 private819.4732us,
NoC801.3744us, direct797.9644us; long-context targets pending. These small controls
cannot establish long-context gains. Preserve earlier cache tuning: VER-02
same-guest T1024 cache is848.8716us before/after protocol update. A target-size
winner replay can test compatibility; SPMCP-only changes do not alone justify
repeating a cache tuning sweep. This is not yet T4096 equivalence evidence.

Before a new launch cite relevant ledger IDs, identify the missing evidence or
changed model/configuration, and check accepted/active/pending cases. Numerical
acceptance, tuning coverage, hardware credibility and full-model applicability
are separate claims. Failed and cancelled portions remain excluded.


**FA-34 retained-profile completion:** All19 cases accepted; experiment_retained_v1/status.json complete, no failures. T4096/Hq14 Q/O-coherent ordinary12391.6680us versus optimistic-ring11802.3292us:4.7559% lower full ROI, but25.2436% slower than same-ELF V-only9423.5016us. This is the fastest tested T4096 ring placement. Q/O-resident ring12773.3000us also loses V-only. Zero transfer-service-time assumption does not establish realizable ring speedup; neither tested ring organization beats V-only. Full acceptance markers retained under results/cacheflex_flash_attention/optimistic_ring_20261001/experiment_retained_v1/runs. No further experiment launched.


**L3-10 retained T16K accepted pair (2026-10-03):** Private142105.2900us and NoC-L3 133777.5472us both accepted in retained_protocol_restart_20261002/long_context_spm_only. Full-ROI latency reduction 5.8603%; speedup 1.062251x versus private. Direct T16K still running, T32K pending. Transferred settings, not independently tuned T16K winners. Values equal historical pair, but these are newly accepted retained-protocol revalidations.

**FA-35 progress (2026-10-03):**110 accepted gates/targets; axial target phase ongoing. Best current private V9418.3056us and L3 9391.4172us remain Br16/Bc256/R256. Crossed parameter and neighbor phases not complete; no final-optimum claim. Current acceptance evidence in current_results.json.


### FA-36 — V-only optimistic rotation (2026-10-03)
User authorized implementing/running V-only rotation. FA-34 already tested K+V
rotation and lost to V-only; FA-35 covers placements but not this schedule.
New kernel retains legacy query-group/core ownership and coherent K/Q/O, rotating
only V. Matched cyclic normal-copy and ring arms separate delivery from schedule;
same-new-ELF legacyV control captures schedule/mapping overhead. V slot mapping
moves to sets768+ for existing hook reuse; disclose this vs baseline sets0+.
Nine gated cases T1024Q2/T1024Q14/T4096Q14; small R128, target winningR256,
Br16/Bc128 or256. Independent numerical/config/work/copy/resource acceptance.
Full ROI must improve matched staging and competitive V-only to support further
finite-hardware work. Three host schedule/source tests PASS; guest compiled.
Reuse frozen retained FA-34 simulator/hook (no relink); no hardware sync added.
Root results/cacheflex_flash_attention/v_only_ring_20261003/experiment_v1.
No accepted runtime result at launch. One new sim under shared20% cap.


**FA-36 T1024 acceptance:** Six gates/pilots accepted. T1024/Hq14/R128/Br16/Bc128 same-new-ELF ordinaryV830.3796us, cyclic-copy1414.2232us, cyclic-ring1343.2696us. Ring reduces matched cyclic ROI5.017%, but is61.766% slower than ordinaryV. Independent full oracle/dense small check, exact cyclic-pair output, copy/rotation/work/config/resource checks passed. T4096 three-arm comparison running/queued; no target result yet. Evidence v_only_ring_20261003/experiment_v1. No finite hardware-sync claim.


**FA-36 T4096 control accepted:** New-guest ordinary V-only9036.8380us accepted full checks; cyclic-copy/ring still pending. This is a different ELF from FA-35 (9418.3056us) and FA-34 (9423.5016us); do not attribute the difference to ring movement, which is disabled in this control, or pool it into their matched ratios. Code-layout/guest sensitivity is not isolated here. Use this same-new-ELF control for FA-36 conclusions.


**FA-36 T4096 cyclic-copy accepted:** Same-new-ELF cyclic-copy14489.4968us versus ordinaryV9036.8380us (60.338% higher latency). Ring target running; no accepted T4096 ring benefit yet. Ring must remove37.632% of cyclic-copy ROI to match ordinaryV. This schedule changes visitation/order, synchronization and V slot placement; no isolated barrier-cost attribution.


**L3-10 retained T16K direct accepted:** Direct133423.6508us accepted vs NoC133777.5472us and private142105.2900us. Same transferred T4096 settings; not independent T16K tuning. T32K private running, NoC/direct pending. Evidence retained_protocol_restart_20261002/long_context_spm_only/runs/t16384_direct/accepted.json.


### FA-37 — DSB coherent/SPM/bridge attribution (2026-10-03)
User authorized counters and rerun. FA-34/FA-36 measured schedule/transfer ROI,
not DSB-specific coherent waiting. This is diagnostic revalidation, not new sync
hardware. Four isolated O3 object replacements (commit/LSQ/rename/memdep) compile
PASS; linking against same retained model and accepted ring hook. Disabled/on
controls must match every old finite nonhost stat and output bit, normalized
config, and full independent oracle. Two OFF controls then allnine FA-36 cases
ON. Counter masks coherent1/SPM2/bridge4 separate drain causes, plus rename
serialization and younger memref dependence counts. Count sampled cycles, not
additive removable walltime; bridge completion remains necessary. Interpret
coherent-only drain alongside rename exposure before proposing narrower sync.
README benchmarks/qwen/fa/dsb_diag/README.txt specifies scope/limitations.
Root results/cacheflex_flash_attention/dsb_diag_20261003/experiment_v1;
no runtime acceptance yet. One sim8GiB lease under shared20%caps; link24GiB,
23GiBAS single thread. Existing queues and frozen model/guest unchanged.


**FA-36 complete:** Allnine accepted. T4096 ordinaryV9036.8380us, cyclic-copy14489.4968us, optimisticV-ring13921.9120us. Matched rotation saves3.9172% but ring remains54.0573% slower than ordinaryV. Same guest comparisons/full checks. Diagnostic FA-37 now measures DSB waiting; no existing result isolates DSB/coherent drain contribution.


**FA-37 T1024 accepted diagnostics:** Both OFF controls and allsix ON T1024 gates/pilots accepted exact every old nonhost stat/output/config plus full independent math; instrumentation is timing-neutral on these cases. Full14head ring912 retired DSBs,27960 sampled commit-drain core-cycles:2714 coherent-only,25246 bridge-only,0 unknown/mixed/SPM-only. Rename serialization88434 core-cycle samples overlaps drain and exposes1300 unique coherent-front instructions (speculative scope); not additive removable runtime. Cyclic-copy drain207509 samples,1686 coherent-only/205823 bridge-only. These small-case data do not support unrelated coherent drain as a dominant slowdown; bridge waits remain necessary. T4096 three ON replays underway. Evidence dsb_diag_20261003/experiment_v1/REPORT.txt and per-case accepted counters.


### FA-38 — V ring scheduling/neighbor/reuse factorial (2026-10-03)
User authorized overlap, neighbor readiness and more query work per circulation.
FA-36 idealized ring remained54% slower than V-only; FA-14 double buffering on
older scaledL2 regressed. New variants reuse coherent K and legacy Q ownership,
not KV partitioning: same-newELF ordinaryV/oldglobal/refactoredglobal/overlap,
neighbor2buffer without/withoverlap, then2head/7headreuse. Bufferfootprint64KiB
atT4096; groupedQ/O coherent footprint grows explicitly. Neighborflags are timed
coherent atomics; localDSBs retained. No hardware SPM-sync instruction implemented.
Optimistic one-hop hook extends isolated pseudo_inst only; realpreclaimedpayload,
zero modeledlinkservice. Main frozen files unchanged. Four host tests PASS,
guest+overlaycompiled. Thirty-case failclosedqueue:2exactcontrols,4directed incl
negative,8modes atT1024Q2/T1024Q14/T4096Q14. Independent fulloracle/exactcyclicbits/
config/work/copy/transfer/resourcechecks. Decision fullROI gain over ordinaryV,
not reduction in stall sums. Detailed contrasts/caveats in
benchmarks/qwen/fa/ring_async/README.txt. Results root ring_async_20261003.
No runtime acceptance yet. Link24GiB,simulation8GiBlease,oneworker,twoCPUs,
shared20%caps and64GiBheadroom. Avoid duplicate queues; status authoritative.


**FA-38 complete (2026-10-04):** All30cases accepted (2exactcontrols,4directed includingexpectednegative,24FA). Same-newELF T4096 us: ordinaryV9054.5944,oldglobalring13926.9088,refactoredglobal15104.5132,globaloverlap14972.2112,neighbor14647.7148,neighboroverlap14399.9368,2headreuse15366.6208,7headreuse19684.9176. Overlap modestly improves each matched implementation; neighbor+overlap still loses oldglobal and ordinaryV. Headreuse regresses despite amortized copies; do not attribute to coherentcapacity alone without counters. No tested ring beats V-only. Maincontrols/config/fullmath/bits/transferbounds passed; zero transfer timing remains optimistic.


**FA-37 complete (2026-10-04):** All11 exactstat/output/config/fulloracle replays accepted. T4096 ring3600retiredDSBs,124832sampled commit-drain corecycles:1445coherent-only/123387bridge-only (~98.84%bridge). Rename422191corecycles overlaps other counts; not additive runtime. Cyclic-copy802520drain samples:957coherent/801563bridge. Evidence does not support unrelated coherent drains as a major ring bottleneck. Counter limitations in README preserved; this does not isolate inter-core barrier waiting or cache effects.

### FA-39 — Winning Qwen V-only counter audit and targeted follow-ups (2026-10-04)

**D; existing accepted outputs only, no new simulations.** Refreshed243 records,
zero discrepancies. Current V9418.3056us versus same-guest matrix cache10432.5348us
is a descriptive cross-protocol/build comparison, NOT new target replay acceptance.
Exact config differences preserved; same tasks/updates/ELF, new disabled-zero
reduction fields. Old tuned same-guest R062/R006 independently corroborate trends.
Current SPM has12.827M vs36.903M L1D misses,1.431M vs2.877M privateL2 misses;
miss-rate increase8.571% vs6.414% reflects smaller denominator. Link traversals
81.336M vs96.126M DECLINE15.4%; ~1.41x is mean packet network latency, not traversals.
SPM loads7.799M, mean service7cycles, issue-to-completion~9; zero RAWwait/reschedules,
13port rejects, LQpeak6/32. SQpeak48/48 with662204SPM-LSQ-full events;
privateL2 creditfull checks7.868M vs0; residual L1D creditfull64.312M vs124.854M.
Instructions+3.405%. PackedK load PCs have11.056M unexecuted-head samples;
packing stack load5.826M; staging UDIV2.196M. Samples/retries are not removable
runtime fractions; network mean depends on message mix. No single-primary-port proof.

Proposed only: exact staging-address specialization (new exact optimization),
isolated8->16L1D/20->40privateL2 credit2x2 (FA-23 transfer to current Qwen, not novel),
direct softmax-to-packed-P (FA-06/29 unimplemented distinction, not directPVepilogue).
Report defines fullROI decisions, correctness/layout controls and matched-cache
requirement for architectural claims; existing SPM-only scope respected, no launches.
Do not repeat FA-21 idealcopy, FA-15–24 fastloads, FA-26forwarding, FA-27earlyaddress,
activeFA-35placement matrix or completedFA-34/36/38rings without new evidence.
[Report](../../results/qwen_v_bottleneck_audit_20261004/REPORT.txt),
[reproducible evidence](../../results/qwen_v_bottleneck_audit_20261004/evidence.json).

**FA-39 implementation/launch (2026-10-04):** User authorized implementing the
three experiments specifically to separate SPM-specific limitations from general
cache/kernel limits. Isolated benchmarks/qwen/fa/bottleneck built one runtime-flag
ELF from matrix guest_v3; retained simulator reused unchanged. Six host tests PASS
(address/layout/work/pairs/parameter scope plus four config-negative checks).
Stage shortcut removes division for rows<=sets, same copies/order/slots. Direct
P producer rounds identically then scatters into eight-row PV layout; scatter cost
is real, so a loss would not rule out other fusion implementations. One ELF keeps
hot code placement fixed across variants; its controls do not replace old winners.
25cases: exact oldprivateT256,12pairedcache/SPM T256 gates,12T4096targets;
control/stage/L1D16/privateL240/both/packed, allBr16Bc256R256Hq14Hkv2causal.
Cache stage flag is a no-op control. Cache runs are attribution controls authorized
by current task, not a cache retuning sweep. Fullmath/densegate/exactpeerbits/work/
copies/config/finitebounds checks, all gates before targets, fail closed.
Acceptance and ROI differences are reported per backend; only differential gains
support SPM-specific attribution. General shared gains remain valid negative evidence
for the SPM-specific hypothesis. Current root
results/cacheflex_flash_attention/bottleneck_20261004/experiment_v1/status.json;
first exactprivate simulation entered, runtime acceptance pending. Coordinator/sim
namespacePIDs2/3 are not hostPID claims; foreground tool session56371 owns lifetime.
One new sim/two allowedCPUs;10GiB lease (8sim+2coord), admitted4GiB growth, shared20%
CPU/RAM and64GiB host spare.120GiBreserved/150.987cap at launch; hostRSS~18GiB,
pressurefalse. Existing FA35/long unchanged. Three prebuild string-count assertions
preserved; none ran simulations. Current-result log SOURCES includes FA39.
First exactprivate control accepted131.656us with all finite nonhost stats and
output bits identical; new-guest cache gate now running. No intervention ROI yet.

**FA-39 validator repair (2026-10-04):** New cacheT256 simulation completed and
passed numerical/config checks but work validator used `(group-head)%groups%8`,
incorrect when groups<8. Correct ownership is
`(head*groups+(group-head)%groups)%8`, directly derived from frozen guest task loop.
T256 expected tasks2/2/2/2/2/2/1/1, updates512/512/512/512/512/512/256/256.
Added independent forward-task-loop test at T256/512/1024/2048/4096;6hosttestsPASS.
Verified all235 frozen inputs before changing only validator/runner/tests. Original
queue/scripts and647 raw run-file hashes preserved at experiment_validation_failure.
Same experiment_v1 resumed with raw exactprivate/cache outputs replayed through all
checks after receipt and exact-command verification; no completed sim repeated,
no guest/hardware/tolerance change. Foreground session4355 now owns continuation.
No intervention target accepted yet. FA35 now170accepted,2active/2pending in
interaction phase; private9418.3056/L39391.4172us winners unchanged. No new best claim.

**FA-39 parallel T1024 screening (2026-10-04):** User explicitly keeps4K queue and
adds1K in parallel. New isolated benchmarks/qwen/fa/bottleneck_1024, same guest_v1/
retained simulator, Br16Bc256R256Hq14Hkv2causal.12cache/private paired cases for
control/stage/L1D16/privateL240/both/packed. Reuses accepted corresponding T256
gates with marker/hash checks; waits for missing gates, no duplicated gate or build.
Command/config/pressure/parameter/work/argv ASTs identical to corrected FA39;
two new hosttestsPASS (pairs plus independent forward-task work). Same fulloracle/
exactbits/work/copies/config/bounds acceptance. Screening is not evidence of the
exact4K bottleneck; compare size interactions after target results. Foreground
session33946, screening_t1024_v1/status.json now authoritative for1K. Existing
experiment_v1 remains untouched and active (7accepted at handoff). One extra sim,
two other allowedCPUs,10GiB initiallease,4GiB admittedgrowth, shared20%CPU/RAM
and64GiBspare. No new result yet. Current log sourceFA-39-1K added.

**FA-39 numerical failure / non-fusion continuation (2026-10-04):** Original
experiment_v1 stopped at gate_cache_packed after11accepted exact/gate cases.
Packed producer outputs all229376 values zero;229350 oracle errors,maxabs0.25.
This is an actual guest/operation correctness failure, NOT a validator-work bug;
no timing accepted, no tolerance relaxed. Scatter producer root cause unresolved.
All235frozen inputs verified unchanged. Isolated bottleneck_targets runner reuses
accepted T256 gates and runs ten authorized pending4K control/stage/L1/L2/both
cache/private cases with same guest/simulator/full checks. Original failed evidence
preserved; packed branch excluded pending separate repair, not dropped as valid null.
targets_continuation_v1/status.json, foregroundsession13396; one sim through same
broker, original1K queue untouched. No duplicate4K target existed at launch.
T1024 fiveaccepted so far: cache997.9288/private935.0756us; stage1001.8576/940.3888us
(-0.3937%/-0.5682% gains): no useful staging shortcut gain. L1D16cache869.6208us
(12.86% lower), private pending; cannot call admission improvement SPM-specific.
Same-newELF comparisons only,1Knot4K attribution. Matrix171accepted, staticKVboth
Br32Bc256R25622192.7624us loses currentVwinner; long32Kprivate remains active.

**FA-39 T1024 non-fusion screening complete (2026-10-04):**10/12accepted; remaining
two packed cases blocked by failed T256 prerequisite (not new simulation failures).
SameELF cache/privateus: control997.9288/935.0756,stage1001.8576/940.3888,
L1D16 869.6208/872.9656,L240997.9288/942.7948,both869.6208/877.9660.
L1D increase improves cache12.8574%,SPM6.6422%; cache slightly overtakesSPM.
L2 increase yields0%cache and-0.8255%SPM gain; combined is worse than L1D-only
forSPM. These results do not identify an SPM-specific staging/admission bottleneck;
L1D sensitivity is stronger in cache. No4K extrapolation, no packed timing accepted.
4Ktargets continuation active firstcachecontrol; matrix172accepted,two final
interaction casesactive before neighborselection;32Kprivate continues. RAM~19.6GiB,
120/150.987GiBreserved,pressurefalse. Originalfailure/screeningstatus preserved.

### FA-40 — Isolated SPM bridge/service headroom series (2026-10-04)

User authorized SPM-specific independent/cache-matched-parameter/idealized tests.
Cross-reference FA39: L2credit relief shifted pressure without improving1K;
FA21 already idealized repeated copies on old single-head64KiBL2; this is a
current-Qwen/protocol transfer, NOT a new functional oracle. FA15–24 service
negatives retained; no duplicate activecase. Original matrix guest_v3 and retained
model reused. Isolated lsq_unit.cc overlay excludes copy shadows from shared48SPM
SQ budget only when env-enabled, replacing it with separate96 or4096diagnostic
credits. All shadow records, original completion/retirement/RAW/WAR/DSB logic,
regularLSQ and pureSPMstore bounds retained.4096 is optimistic metadata headroom,
not area/feasibility claim. New counters record actual shadow/totalmax and sampled
admission relief, not exposed stalls. Default exact-stat/output gate required.

Independent arms: bridge96/4096, copy-request4->2 (ordinary request parameter only),
existing ideal repeated-copy oracle, SPMresponse5->3/1, loadports1->2/8; combined
bridge4096+oracle+response1+ports8. Cache combined must be exact stats/output no-op.
Instruction issue costs remain: FA39 stage-address negative reused, not mislabeled
as zero-cost CPU submission. Matching isolated parameters is not entire cache-path
equivalence. Full copy-instruction elimination is not implemented in this series.

38cases: exactprivate,8directed roundtripRAW/release gates,12T256 and12T1024 cases,
5T4096 cache/private controls plus bridge4096/oracle/combined confirmations.
Fulloracle/exactpeerbits/work/copy/config/bounds and oracle-exercise checks; failures
stop acceptance. Four hosttestsPASS incl unchanged ordering-code ranges; isolated
lsq_unit.o compiles. Linking under24GiB lease,23GiBAS,singlethread; foreground
session43214 automatically proceeds to queue after link. No runtime acceptance yet.
Root results/cacheflex_flash_attention/spm_isolation_20261004, source
benchmarks/qwen/fa/spm_isolation/README.txt. Existing runs/binaries/sources untouched.
Compile4GiB, link24GiB, then one sim/two allowedCPUs/10GiB initiallease with admitted
4GiB growth; shared20%CPU/RAM and64GiBhostspare. At link120/150.987GiBreserved.
Decision fullROI: substantial improvement plus targeted relief identifies causal
headroom; counter relief without speedup demotes that resource. Bundled gain needs
individual-arm interpretation and is not additive per-resource runtime attribution.
Link completed SHA50fed894f2351698ec283886cc40d5fafe70379ba9b96b61b4aff494d7b98589;
exactprivate runtime control entered without launch errors. Current-result source
FA40 added; runtime acceptance pending. Source/ordering changes remain isolated.

**FA-40 small gates complete (2026-10-04 status):**21/38 accepted: exact default,
8 directed gates and12 T256 cases. T1024 cache control now running; no target-size
conclusion yet. T256 SPM control131.6560us; bridge96/4096 both131.5356us (0.09145%
lower), copy-request2 131.6788us, copy oracle127.7668us (2.9541% lower), combined
127.6880us (3.0139% lower). Load response3/1:131.6488/131.6560us; ports2/8 both
131.6560us. Cache combined125.3088us exactly matches cache control stats/output.
Bridge expansion removes SPM LSQ-full events31573->0 without meaningful runtime
gain; ideal repeated-copy delivery helps modestly at this small gate,3072/7168
copies eligible, first claims/fallback retained. This is not free CPU submission,
not full cache-equivalence, and not evidence of the dominant T4096 bottleneck.
FA39 continuation1/10 accepted: new-guest4K cache control10484.1404us, private
control running. Do not combine with original-matrix guest timings. FA35 has173
accepted, one static-KV interaction active before neighbor selection; latest
KVboth Br32Bc128R64 23051.7752us does not improve V winner.32Kprivate still active.
Current log refreshed291accepted/zero discrepancies; active queues report no new
errors or memory pressure. Earlier packed-producer correctness failure remains
preserved and excluded, not a valid negative timing result.
Immediate FA35 completion:174accepted; final staticKVboth Br32Bc128R256
23058.2248us. Interaction phase complete, winner_neighbors_gates entered;
T1024headQO Br64Bc128R128 acquired lease. Thus neighbor checks now beginning,
not a frozen final optimum. Refreshed current log after this completion.

**Status follow-up 2026-10-04:** FA40 22/38accepted: T1024cachecontrol997.0628us;
cachecombined no-op validation running, no new SPM comparison yet. FA35 175accepted:
T512headQV Br32Bc128R64 gate394.5364us accepted; T1024headQO Br64Bc128R128
neighbor gate active. Small gate is not a target winner. FA39 still1/10,4Kprivate
control active;32Kprivate active. Four queues have fresh heartbeats,no new errors;
RAM22.9GiBactual,106/150.987GiBreserved,pressurefalse. Current log294accepted,
zero discrepancies. No changed target-speedup conclusion.

**Status follow-up 2026-10-04,300accepted:** FA40 26/38accepted; T1024 cache
control/combined997.0628us exact no-op; SPMcontrol959.2636us, bridge96/4096
both956.9056us (0.2458% lower). Queue expansion alone again yields little benefit;
copy-request2 active, oracle/service/combined screening pending. FA39 2/10accepted:
matched new-guest T4096 cache10484.1404/private9449.2652us =1.10952x,9.8709%
lower latency. This is control comparison, not benefit from a diagnostic change.
Cache staging-address variant active. FA35 176accepted; T1024 QO Br64Bc128R128
gate985.5748us accepted; two4K QO neighbor targets active. No new target winner.
32Kprivate continues. Five active simulations across four queues, fresh heartbeats,
no errors or pressure;44.5GiBactualRAM,130/150.987GiBreserved. Current log300
accepted/zero discrepancies. Earlier invalid packed branch remains excluded.

**FA40 T1024 copy results (2026-10-04):**28/38accepted. Copy-request2
960.2992us versus959.2636us control (0.108% slower); ideal repeated-copy oracle
951.4252us (0.817% lower). Together with bridge-only0.246% benefit, these demote
copy transport/shared shadow capacity as dominant removable T1024 costs under
these interventions. They do not remove CPU copy instruction issue, first claims,
or all staging work; do not conclude staging is free or fully ruled out. Faster
read/service arms and combination still pending atT1024, confirmations atT4096
pending. No unique dominant SPM-specific bottleneck established. Current log302
accepted/zero discrepancies; FA39 target unchanged2/10.

**FA40 read-latency screening / architecture interpretation (2026-10-04):**
T1024 response3 accepted955.1828us vs959.2636us control (0.4254% lower);29/38
accepted, response1 running. Demotes modest read-service latency as dominant at
this size, not a proof of CPU arithmetic saturation. FA39 L1D16 causal gains
12.86%cache/6.64%SPM remain strongest demonstrated shared constraint. MOD11 PV
engine analytical evidence reused; no new experiment proposed/queued here.
SPM-resident vector FMA is a design hypothesis: ownership/residency can simplify
operand handling, but current results do not establish full-ROI gain or area
advantage over coherent near-cache execution. PrivateV locality must be respected;
L3 placement adds operand-transfer costs. Current log303accepted/zero discrepancies.

### FA-41 — Llama3.2-1B FA evidence-guided baseline tuning (2026-10-04)
User authorizes implement/run/review/iterate bothcache/privateSPM, applying Qwen
lessons without placementmatrix. New shape32Q/8KV/d64,T4096,B1,8cores; exact frozen
matrixguest_v3 already supports shape, retainedgem5 unchanged. Prior FA30 was
llama.cpp reference NOT same model/precision; FA35/L3-11/12 justify initial
Br16Bc256R256, FA39 counters order hypotheses; MOD12 timing rank not used.
RollingV32KiB initial, cache matched directPV kernel. All arithmetic/copy/work/
config/bounds checks preserved. Dense smallgate+independent fulloutput oracle;
newBc separate reference and exactpeerbits only sameT/Bc.201dependencies verified.
Five hosttestsPASS including workpartition, no duplicateadaptivecases, invalid
geometries, and config negatives for changedcapacity/guest/argv.

benchmarks/llama/fa/iterative/run.py performs one-change local iterations, each
chosen from accepted currentwinner+counters+exact before/after work/copies.
Br16/32,Bc128/256/512,R128/256/512; max8targets/backend, no Cartesian product or
otherplacement tuning. Gates precede targets. Neighborhood convergence is local;
cap stops unresolved, final awaiting_final_review never automaticpaperready.
All targets and negative results retained; final review may motivate placement
pivot or code change rather than unconditional freeze. FullROI reduction and
predictedcounterchange test each hypothesis; samples not additive timefractions.
Scope is FP16-GEMM/FP32-state syntheticFA, not modelaccuracy/endtoendLLama.

Root results/llama_fa_iterative_20261004/experiment_v1. One sim,2allowedCPUs,
shared20%CPU/RAM,64GiBspare; idlecoordinator0.5GiBlease/AS, upgrades atomically
into10GiBlease(8sim+2coord),sim admitted4GiBgrowth. Current existing reservations
150/150.987GiB: queue waits for capacity, no oldjobpause/restart/cancel. Launch
pending this entry; no Llama runtime result yet. Model/scope documented at
results/llama_fa_iterative_20261004/MODEL_AND_METHOD.txt.
FA41 launched foreground toolsession95872, namespacePID2 (not hostPID). Fresh
status waiting_for_reservation, zeroaccepted. Idle0.5GiBlease makes150.5GiBtotal
under150.987GiBcap; automatically upgrades when10GiB fits. After first simulation
coordinator retains2GiBidlelease/AS for oracle/JIT mappings; no unsafe shrink.
Current-result source added. All existing queues retain their workloads/PIDs.
FA41 status follow-up: still waiting, noaccepted. FA35 both current QO neighbor
simulators nowpaused for admitted-memory growth, leases36/32GiB, unchangedPIDs.
FA40 targetcache and FA39 targetSPMstage unpaused; long32Kprivate queue reports
running.150.5/150.987GiBreserved, actual75.96GiB,hostpressurefalse. Reservation
limit differs from actualRSS. No newfailure or launch made in statusreview.

**L3-10 cancellation requested, host action blocked (2026-10-04):** User says
kill qwen long context. Tool namespace cannot see host2387013/2389369/2659595;
SSH socket denied by environment. NO host signals sent, NO lease released or
cancelled status fabricated. scripts/cancel_qwen_long_host.py prepared and syntax
checked, read-only preflight fails safely on missing host /proc. Host execution
required. Parent has fail-all exception, so helper stops original parent before
killing longworker/current sim, preserves results, releases only retained_long,
and replaces original resource heartbeat/pressure policy. Cancellation is NOT
complete until host cancelled.json exists and verifies dead processes. Other
queues continue; FA41 remains queued under existingcap.
Cancellation helper host attempt failed BEFORE signals: parent cmd uses relative
results/.../run_spm_only.py, old check expected absolute. Preserved failurelog and
script in results/cancel_qwen_long_20261004/path_check_failure.*. Fixed identity
check resolves argv paths against actual /proc/PID/cwd and checks exact canonical
script or intended output subtree; no basename-only relaxation. Six positive/
negative identitychecksPASS (relative/absolute,wrongscript,cwd,outputqueue).
Still NOT cancelled, retained_long56GiB lease present; FA41 fresh waiting.
User must rerun corrected host command because hostsignal access still unavailable.
Host cancellation retry again failed BEFORE signals: systemPython lacks
os.pidfd_open. Preserved python_api_failure.log. Helper now uses pidfd when
available, otherwise UID+exactcmd+starttime recheck immediately before os.kill.
Fallback actual signal test on owned disposable sleep process PASS; wrongstart
rejected, relativepath stillPASS. Hostlong NOT yetcancelled, leaseunchanged.

**User cancellation L3-10:** host script cancelled retained long queue and current32K simulation; accepted records and partials preserved. Original parent 2387013 intentionallySTOPPED, NEVERresume; monitor 635092 now owns shared heartbeat. Released retained_long lease only. Evidence results/cancel_qwen_long_20261004/cancelled.json. Other queues untouched.
Host cancellation verified: cancelled.json exists, longstatuscancelled_by_user,
retained_long56GiBlease removed, replacementmonitor635092 fresh. Six accepted
longresults preserved. FA41 automatically admitted firstT256cache32/8gate,
namespacePID4 unpaused, zeroaccepted. BothFA35neighbors resumed admittedgrowth.
Reservations112/150.987GiB,hostpressurefalse. Neverresume originalparent2387013.
FA41 firstLlama gate accepted: T256cache244.1948us, all524288outputs verified,
zeroerrors/nonfinite, denseFP64 and exactwork/config/bounds passed. T256SPM
currentlyrunning, no T4096Llama timing yet. ETA estimate only: matched Qwen
4K hostdurations44–49min; Llama32/14head work scaling suggests~1.5–2.5h per
4Kcase allowing largerKVset/export effects. Not a measured Llama target duration
or speedup. Full local search remains queued, not replaced by manual review.

**FA-35 completed:** 180 accepted gates/targets; best measured per-family results: /data2/rsappidi/cacheflex_mc/results/retained_protocol_restart_20261002/fa_matrix_spm_only/winners.json. No global-optimum claim.

**FA41 initial target pair accepted (2026-10-04):**5accepted incl bothT256controls,
T4096cache/SPMBr16Bc256R256 and cacheBc128gate. MatchedLlama32/8/d64 target
cache23499.3756/private21296.3536us =>1.103446x,9.37481%lower latency. Hostwall
132.57/119.68min includingvalidation. Provisional transferredgeometry, nottunedfinal.
SPM instruction918.916M->950.175M;L1miss86.638M->27.476M;L2miss6.695M->3.476M;
L1creditfull300.534M->154.177M. Copy/pressureoffset remains, countersnotstallfractions.
Automatedlocalrunner nowcacheBc128target, unchangedBr16R256; no placementpivot.

FA35complete180accepted, winnerVBr16Bc256R2569418.3056us unchanged; bounded
screenedmatrix notglobaloptimum. FA39 now8/10accepted:T4096stageSPM9225.5676us
vs9449.2652control (2.36735%lower), cache10435.3332vs10484.1404 (~0.466%lower).
L1credits16 cache8744/private8616.7168us =>16.5978%/8.81072%lower respectively;
sharedadmissionlimit, strongerbenefitcache. L2credits40 cache10484.1404(private
9541.2396) unchangedcache/regressiveSPM. Bothcreditarms stillpending. FA40 34/38,
4KSPMcontrolactive; prior1Kcombined948.182vs959.2636us (~1.155%lower) not4Kclaim.
Currentlog323accepted/zero discrepancies. Three simsactive, noerrors/pauses;
36/150.987GiBreserved,~5.32GiBactualRAM. Longqueue remainscancelled, monitor635092.

**FA41 coherence/traffic audit:** same-guest/simulator T4096Br16Bc256R256 accepted
cache/SPM initialpair, statsSHA andmarkers verified. NoCpackets26,387,495->21,638,694;
injectedflits62,790,123->53,985,726;linktraversals238,281,353->201,628,088.
PrivateL2controller (SLICC L1Cache_Controller) Inv transitions1,296,303->872,459;
Fwd_GETS304,425->196,972;Fwd_GETX1,242,801->568,349. These establish reduced
coherence-controller activity alongside reducedTOTALnetwork traffic, not pure
coherence-byte reduction or unique-message counts. SLICC countTransition increments
on valid transitions; do notsumstateaggregate/perstate or controllerlevels.
Evidence results/llama_fa_iterative_20261004/coherence_audit/evidence.json.
Read-only analysis, no experiment or inputchange; not scaling/NoCsaturation proof.

**Historical Qwen packet-latency decomposition (2026-10-04):** Read-only FA39
follow-up on spreadsheetR062cache/R006private same-geometry historical pair,
not mixed with retainedLlama. Artifact results/packet_latency_audit_20261004/
{analyze.py,evidence.json}; firstROI typedmessages sums exactly reproduce total
receivedpackets and mean networklatency. Mean network5.238494->7.426247ns,
queue5.776520->6.409433ns. resp_DATA mean8.721->15.677ns; privateL2->privateL2
9.611->10.154ns, sharedL3->privateL2 8.343->17.271ns. L3->L2DATAcounts1,885,955
->1,839,108; itsqueue12.977->13.133ns. DATA5flits/packet both; networkmeanhops
1.741602->1.771895 (+1.74%). Therefore not only removalofcheapmessages or larger
DATApackets: responseclass latency genuinely rises, concentratedL3->privateL2.
TypedDATAweightedcontribution accounts+1.9884ns of+2.18775ns globalnetworkmean
change (descriptive, NOT runtimeattribution). Aggregate routing/type mix within
class and burst contention not separated; no exactrouter/port causalproof.
NetworkInterface measures tail-flit networktransit; queue includes message-source
waiting and destination wait afterarrival, NOT justrouterqueue or DRAMlatency.
Potential stagingbursts/hotspots/flowcontrol remain hypotheses; totaltraffic
reduction doesnotestablishlower peakcontention or NoCsaturation. No newrun.

**Historical packet route follow-up (2026-10-04):** R062/R006 read-only
per-source/destination DATA analysis saved at results/packet_latency_audit_20261004/route_evidence.json. All64 sharedL3-bank->privateL2 routes slower in SPM. Reweighting cache route means with SPM packet counts gives8.343771ns versus cache8.342797ns, while actualSPM17.271129ns: route mix explains only0.000974ns. Example bank0->core6 10.181823->31.022629ns, bank7->core1 10.002721->35.668134ns. This excludes route-mix-only explanation, not temporal/message-subtype mix. Source audit clarifies network latency starts at flit creation/enqueue in NI output VC, BEFORE NI arbitration/credit-gated send; includes source NI VC waiting as well as router/link traversal. Thus previous wording networktransit must not imply router-only time. Cannot isolate NI arbitration versus router credit/backpressure or assign DATA delay specifically to SPM copy replies from aggregate counters. Staging burst interference remains hypothesis, no new simulation or causal claim.

### FA-42 — Packet-path diagnostic isolation (2026-10-04)

User authorized experiments to isolate increased packet latency. New code
benchmarks/qwen/fa/packet_diag; result root
results/cacheflex_flash_attention/packet_diag_20261004. Six host tests PASS
(interval accounting/classification/censoring/negative config/paired plan).
Isolated NetworkInterface overlay adds observational message classification,
per-ready-VC credit/order eligibility and receiver stall labels; existing router
trace events reused. No modeled timing/resources, source protocol or guest changes.
Build launched tool session63185; runtime acceptance pending.

Prior FA-13/16/18 used older-profile selected traces; FA-15/20 altered network
bandwidth for all traffic; FA-21/40 idealized several copy costs together. New
queue revalidates exact T256 accepted private control with trace off/on, then
replays FA-39 retained-protocol accepted T4096 cache/SPM controls at early and
40%-ROI windows, off/on paired bounded stops. Ten serial cases, two full small
gates plus eight prefixes, NOT a retuning or duplicate full4K baseline sweep.
Does not mix historical R062/R006 timings with new retained controls. Each10us
window reports complete/censored packets, copy/coherent DATA, tail-flit NI wait,
router pipeline/wait, crossbar/link time and receiver ejection wait. Credit/VC
block counts are samples, not additive runtime fractions. Source queue wait
overlaps message age; no stall sums. Strict config/source hashes, exact small
output/all finite nonhost stats and prefix trace-on/off stats are required.

A repeatable localization will justify selecting a single follow-up NI/router/
receiver intervention; no such performance benefit inferred automatically. A
full-ROI outcome changing the design decision must show meaningful SPM runtime
benefit versus matched original and coherent-cache sensitivity control, with
unchanged arithmetic and all other model parameters. Prefixes cannot do this.
Resources share20%CPU/RAM broker,2allowedCPUs/1sim;6GiBcompile,24GiBlink,
10GiBsimulation lease (8GiBinitialAS),admitted4GiBgrowth,64GiBhostheadroom,
2GiBcompressed trace cap. Other queues untouched. Current-log source registered.

FA42 pre-runtime plan refinement while isolated link pending, before inputs.json
freeze: small exact gate traces its whole131.656us ROI to guarantee copy coverage;
4K windows remain10us each. No runtime experiment yet, no frozen input edited.

FA42 isolated compile/link PASS; ELF4bbdfeb80b3d4e556a4a45fbf7af260dc3a02a9816fa05ad7cfc2126e52e01e0. First exactT256 trace-off full gate active; no runtime acceptance yet. Sources frozen in experiment_v1/inputs.json. Parser v1 censored_packets includes non-Request/Response structs lacking NI_CLASS metadata as well as edge-incomplete packets; not a pure edge-censor count. DATA classification/decomposition unaffected. No active frozen source changed.

**FA42 first runtime gates accepted (2026-10-04):**2/10accepted, exactT256
traceoff/on all finite nonhost stats and output match original.145599 complete
classified packets/145601 created in smallROI trace;6259copyDATA. CopyDATA mean
source-ready wait39.545ns, NI tail wait7.141ns (includes serialization), router
ready-to-grant5.907ns, crossbar/links4.059ns, endpointwait0ns. VC/credit blockage
samples present. These are within-SPM T256 diagnostic samples, NOT matched
cache4K latency attribution or additive runtime fractions; receiver backpressure
can still propagate upstream. Early4Kcache off-prefix nowactive,8prefixes pending.

Status326accepted/zero discrepancies. FA39-4K9/10: cachebothcredits8744us equals
cacheL1-only8744us, so extraL2 credits add no benefit on this arm. SPMboth still
active. FA40 remains34/38,4KSPMcontrol active. FA41 remains5accepted,4KcacheBc128
active. Four simulations running, no newfailures/pauses; fresh hostmonitor
actualRAM6.79GiB, reservations46/150.987GiB,pressurefalse. No newhardwarelaunch.

**FA42 scheduling-only parallel handoff (2026-10-04):** User authorized more
concurrency. New benchmarks/qwen/fa/packet_parallel/{launch,worker}.py,3hosttests
PASS. Tool session89705, authoritative packet_diag_20261004/parallel_v1/status.json.
Three accepted records reused (smalloff/on gates + earlycacheoff), current
earlycacheon preserved. Six remaining earlySPMoff/on,middlecacheoff/on,middleSPM
off/on nowrunning in parallel; trace-on acceptance waits matched off acceptance.
Original scientific validation block reused verbatim with AST hash; model/guest/
commands/windows/stops/config/math checks unchanged. All original2376 frozen
inputs checked. Pendingcase directories atomically claimed with delegated.json;
originalserialrunner's next exclusive mkdir at early_spm_off intentionally raises
FileExistsError AFTER its active case completes, before duplicate launch. This
expected schedulerhandoff is NOT simulator/correctness failure. Original status
and error retained; newaggregator authoritative, currentlog source updated. No
cross-namespace signals and no restarting active simulation. Never resume original.

Peak10simulators:7FA42 +FA39/40/41. Six newworkers use distinct allowedCPU pairs
(indices2..13of24CPUset); parent2GiBreservation, each10GiBlease/8GiBsimAS, admitted
4GiBgrowth; total108/150.987GiBreserved atlaunch, nohold/pressure. Same20%CPU/RAM
and64GiBextraheadroom. Schedulerthroughput improvement only, no new timing claim.

**Status review (2026-10-04, 18:22 PDT):** Current log335accepted/zero discrepancies. FA42 parallel diagnostic queue completed10/10; seven additional accepted prefixes since last review. Exact observer checks passed; detailed matched-window interpretation pending, no new full-ROI speedup claim. FA39-4K completed10/10: SPM bothcredits8726.516us vs control9449.2652 (7.649% lower), but slower than L1-only8616.7168 by1.274%; extra L2 credits add no improvement. Cacheboth8744us equals L1-only. Thus credit sensitivity is shared with cache, not evidence of an SPM-specific L2-credit bottleneck.
FA40 remains34/38 and FA41 fiveaccepted in saved status. After client permission/session change, their status/log heartbeats are several minutes old and previous foreground tool session95872 is unavailable; visible host process list contains no live gem5. Namespace/container absence cannot establish termination conclusively. Treat these two as previously running, live continuation UNCONFIRMED; no restart/duplicate launched. Broker retains two10GiBleases plus6GiBbase (26GiB), host monitor pressurefalse,~2.36GiB visible userRSS; this may omit unavailable tool-container processes. Long context remains user-cancelled.

**FA39 retained-protocol flit latency audit (2026-10-04):** Matched accepted target_cache_control/target_spm_control in bottleneck_20261004/targets_continuation_v1, first stats blocks simTicks exactly reproduce accepted10484.1404/9449.2652us; simFreq1e12. Mean flit total13.611319365->17.642540312ns (+29.617%), network5.387033302->8.961920778ns (+66.361%), queue8.224286063->8.680619534ns (+5.549%). Thus elevated flit latency persists under retained-source protocol, not solely an old-release artifact. Aggregate means are traffic-weighted; do not infer every path or SPM-only packet causal attribution. Network includes NI VC waiting; no additive kernel-runtime interpretation. Read-only audit, no new experiment.

**FA42 matched-window interpretation (2026-10-04):** Four accepted T4096 trace windows localize added SPM-copy DATA tail latency predominantly to router ready-to-grant waiting, with NI waiting secondary. Middle10us window L3->privateL2 cachecoherent2328packets: NI2.694/router2.844/crossbar-links3.820ns; SPMcopy1024packets: NI4.206/router17.655/crossbar-links3.750ns; SPMcoherent981packets: NI2.576/router1.896/links3.745ns. Destinationejectionwait0 for these classes. Copy all-flit blockage samples21059NO_VC+21840NO_CREDIT demonstrate flow-control contention, not additive time nor proof of ultimate downstream resource. Earlywindow cachecoherent5243 NI5.046/router3.556/links3.736ns; SPMcopy4057 NI6.010/router6.880/links3.753ns; SPMcoherent4464 NI4.339/router4.716/links3.805ns. Thus delay varies with traffic phase and is particularly large for copy replies; ordinary coherent replies need not all slow. Tail-packet decomposition explains location, not directly whole-ROI flit mean or removable runtime. Windows are elapsed-offset/proportional-ROI selected, not exact-work aligned. No inherent higher wire latency or receiver post-arrival stall observed; zero ejectionwait does not exclude downstream backpressure. Full-ROI causal bottleneck magnitude remains unestablished. Evidence experiment_v1/runs/{early,middle}_{cache,spm}_on/packet_analysis.json.

### FA-43 — Data-VC depth causal sensitivity (2026-10-04)
User authorized prioritized network-latency intervention atT1024. New four parallel
arms cache/SPM x dataVCdepth8/16 vs accepted4; each T256 gate then fullT1024.
No simulator rebuild, no changes to clocks,links,routing,VCcount,misscredits,
finite-controller resources or kernel. Existing FA39 T256/1024 controls reused
with frozen config/commands/stats/output/acceptance hashes; valid controls remain
valid despite later unrelated packed-kernel failure in historicalqueue.
FA42 copyreply routerwait localizes question; FA15/20 oldglobalbandwidth and
FA21/40 broadcopycost do not isolate depth. Inventory querybuffers_per_data_vc
found no prior matching experiment. Exact config and independent fulloracle,
work/copy/occupancy/no-reduction checks retained, plus exacthistoricaloutputbits.
ThreehosttestsPASS inclwrongdepthrejection; allfour actualruntimeconfigsPASS.
Code benchmarks/qwen/fa/network_relief; root network_relief_20261004 underFAresults.
Fourcoordinators1074954/55/56/57; fourgatesactive atlaunch, noacceptanceyet.
Each10GiBlease/8GiBAS,controlledgrowth;8allowedCPUs,66/150.987GiBtotalreserved
including retainedFA40/41leases; shared20%CPU/RAM,64GiBheadroom. No oldqueue
modified/restarted. Newhostdetachedworkers directlyvisible; statusfiles authoritative.
Currentlog fournew sourcesregistered. BaselineT1024 cache997.9288/SPM935.0756us;
flitnetwork5.683594/7.788345ns, so the latency gap exists atscreeningsize.
Decision >=2%SPMfullROI reduction withpositiveexcessovercache and reducednetwork
latency motivates4Kconfirmation; null/regression preserved. Not precise additive
attribution or proof of4K result. VCcount andcopyoracle followups contingent,
not implemented/queued by this first stage. Increasedbufferstorage disclosed.

**FA43 overlap audit clarification (2026-10-04):** User challenged redundancy.
Broader source search found FA24 scripts/fa_headroom_profiles.py explicitly set
dataVCdepth16 AND controlVCdepth8, together with1024bitlinks, enlargedports/LSQs/
misscredits, lowercontrollerlatencies and greatertransitionthroughput. Accepted
headroom_2x2_v2_20260923/profiles_snapshot.py confirms prior depth16 inclusion.
Thus do NOT claim larger network buffers were never tested. FA43 novelty is
isolated dataVCdepth4->8/16 with allotherparametersfixed on retainedQwen1024,
not discovery of generic network/memory headroom. FA15/20 priorwidening and
FA24bundle already establish common memory headroom, not SPM-specific gains.
Original inventory exact-string query did not surface parameter hidden in bundle;
source-level search required. Current fourT256gates accepted; targetscontinue.
No new launch/retuning proposed in this review; bounded FA43 remains authorized.

**FA43 completed (2026-10-04):** All8cases accepted (fourT256gates+fourT1024
fullROIs), nofailures. Both depth8 and16 produce identical timings/counters:
cache989.5348us vs997.9288control (-0.84114%); SPM934.8004 vs935.0756
(-0.02943%). SPM/cache benefit1.067217->1.058552x. SPMflitnetwork mean
7.788345->7.815520ns (notrelieved), queue8.633353->8.431593ns, total
16.421698->16.247113ns. Cacheflitnetwork5.683594->5.603728ns, total
13.644200->13.494610ns. Therefore additionaldataVCdepth alone doesnot
meaningfully improveSPMruntime; cachebenefitsmore. Does NOT establish network
latency unimportant because targetedSPMnetworkmean didnotdecrease. No automatic
T4096 escalation warranted by declared positivegaincriterion. VCallocation/
copy-specific oracle stillunimplemented; no broad no-network-bottleneck claim.
Allnewleasesreleased, broker26GiB includingtwohistoricalunconfirmedleases.
FA40/41heartbeatsremainstale; do not report them live. Currentlog343accepted,
zero discrepancies. Artifacts network_relief_20261004 fourarm status/accepted/
stats/config and per-armanalysis.json.

**FA42 downstream blocking audit (2026-10-04):** Read-only fourwindow analysis
results/packet_blocking_audit_20261004/{analyze.py,REPORT.md,*_on.json}. Middle
1024copyreplies=512each tocores4/6,128eachfrom8banks; destinationlink100%
100ns-bin utilization vs26.36/27.04%windowaverage. Cachecorrespondingpeaks
25/21.5%; windowsnotworkaligned.99.1%confirmedoccupiedVCsamples oncopyNO_VC
belongtoothercopyDATA;15578/21059samples allfourresponseVCs provablyoccupied
(tailnotyetdeparteddownstream).95.5%co-grantsduringcopytailwaitothercopyDATA,
excludingownpriorflits; countsnotruntimefractions. Destinationports4/6contribute
31.4%copytailrouterwait. Samepacket headNO_VC atdestination andtailNO_CREDIT
upstream observed(e.g7522829tick24432416000router0->4). Copyfan-in bursts and
backpressure areobserved, notcontinuousNoCsaturation. DeeperVCbufferscannot
increaseoutputlinkrate/numberofpacketVCs; plausibleFA43null explanation but
FA43T1024vsFA42T4096notcausalpair. Creditreturnuntraced, arbitrationeligibility
notexplicitlysampled; distinguishfromprovenoccupancy. Portbound/routeconsistency/
independenttailwaitreconciliationPASS. OneCPU2GiBAS,nonewrun/modelchange.

### FA-44 — Paced and overlapped V-only kernels (2026-10-04)
User authorized implementation of paced/overlapped kernels afterFA42 burstfan-in
andFA43 buffernull. PriorFA14 already tested older singlehead async/doublebuffer
andregressed; this is targeted current retainedQwen transfer, notnovelconcept.
Newisolatedguest built fromFA39guest; all37parentfrozenfiles verifiedunchanged.
benchmarks/qwen/fa/paced_staging; results/cacheflex_flash_attention/paced_staging_20261004.
OneELF: cache,originalburst,paced,overlap,doublebufferburstcontrol. Pacedissues
16Vrows/32copies between eachfirstblocksoftmaxrow, beforefirstPV. Overlapstartup
paced, then16nextVrows aftereach16queryrowPVblock,16batches across256queryrows.
Nexttilecausalguard, alternatingdisjointslots sets0..255/256..511 ways0/1.
Vfootprint32->64KiB foroverlap/dbburst; samedoublebufferburstcontrol isolates
footprint/mapping, althoughstartup differs. Existing RAW/WAR/shadow ordering,
no addedfences/untimedcopies/newhardware; finalrelease coversbothusedpanels.
Exacttaskownership,copybytes,arithmeticorder,QK/PV/softmax andoutputs retained.
T256gates thenT512reusegates thenT1024fullROI forfivearms (15newcases). NewELF
controls revalidation necessary forcode-layout effects, historicaltimingsnotpooled;
T256/1024historicalexactbits reused, allsizesfullindependentoracle/work/copies/
finitebounds/noreduction/config/hash/export checks, cross-peeroutputverification.
FourhosttestsPASS(lifetime/copyvolume/slotencoding/geometry/sourcehooks),
fiveactualconfigsPASS, negative buffer/schedule/guest configsrejected. Runtime
pending atlaunch. Coordinators3637051..55, fivegate simsactiveunpaused.
Each10GiBlease/8GiBinitialAS, controlledgrowth;10allowedhostCPUs,76/150.987GiB
reservations including26GiBexistingbase/leases,20%RAM/CPU+64GiBspare.
Originalqueues/model/guestsuntouched. Fivecurrentlog sourcesregistered.
FullROI own/control reductions andnetworklatency decidewhether schedulinghelps;
aggregateflitmeansalonecannotproveburst suppression. Boundedtracefollowup only
if useful; no 4K autoextension or hardwareclaim. Negativeoutcomespreserved.

FA44 first runtime gates: allfiveT256 PASS fulloracle/exacthistoricalbits, config/work/copies/bounds. cache129.2568/burst136.4968/paced132.7268/overlap132.5748/dbburst135.7288us. Validation-size only; T256cannotexercisefuturetile overlap. AllfiveT512reusegates active, zero failures; noT1024 resultyet. Hostmonitor~5.67GiBuserRSS, pressurefalse.

FA44 T512 reuse gates allfiveaccepted: cache368.2464/burst380.86/paced361.0864/overlap362.0268/dbburst368.2004us. Paced5.1918%lower vsburst; overlap4.9449%lower vsburst but1.6767%lower vsmatchedDBburst. These are validation-size results only, notT1024/4096 conclusions; doublebufferalonehelpshere. AllfiveT1024targets active withfreshheartbeats,zero failures/pauses. Currentlog353accepted/zero discrepancies.

**FA44 complete T1024 (2026-10-04):** All15 accepted (10gates+5targets),zero
failures. SameELF T1024cache1022.2184/burst960.816/paced947.9052/overlap958.654/
dbburst957.6892us. Paced1.34373%lower vsburst; overlap0.22502%lower vsburst
but0.10074%slower vsmatchedDBburst. SPM/cache1.0639065xoriginal->1.0783973x
paced. Flitnetwork ns cache5.590660/burst7.889297/paced6.534554/overlap6.557739/
dbburst7.786794. Paced networkmean17.17%lower butfullROIonly1.34%lower;
overlapreducesnetworkmeanwithoutmeaningfulruntimegain. Schedulingoverhead/
changedoverlap preventexactadditive attribution; no substantialrecoverable
runtimeclaim, no4K extrapolation. SmallT512~5%gains didnotcarrytoT1024.
No further launches; allFA44leasesreleased. Currentlog358accepted,zerodiscrepancies.

### SYN-01 — Accepted evidence audit of further private-SPM headroom (2026-10-04)
User requestedcomprehensivevalidhistory/countersearch,nonewsimulations. Report
results/cacheflex_headroom_audit_20261004/REPORT.txt; reproducibleanalyze.py and
evidence.json. Searched594studyroots/4693indexedrecords (notallvalid),checked
358currentacceptancemarkerhashes; selected26currentrawROI/outputhash/arithmetic
acceptance/config/guest/simulatorchecks. Fourprimarycache/SPMpairs sameELF/binary
andonly13expectedmode/way/outputpathdifferences. Notanindependent rerunofall
numericaloracles norparsingeverylegacytrace. H/invalid/failed/partialexcluded.
FreshFA39matched4K:missesL1-64.34%,L2-47.73%,LLC-28.27%,flits-12.29%;
SPM7.799Mloads7cyservice,zeroRAW/reschedules,10portblocks,LQpeak6/32.
PendingPCscoherentpackedK6.203M+5.189M,packing5.997M,normalization2.731M,
stagingUDIV2.000M; samplesnotadditivefractions. L1creditinterventionnarrows
1.10952x->1.01477x; strongercoherentpathhelpscachemore. CurrentFA40smallgains,
FA43null,FA44pacing1.344% andfullFA35placementnegatives demotelargehidden
SPM-specificservice/port/admission/network tax. Doesnotprovecompute-bound.
Nohard1.1xceiling:FA39addressspecializationmatchedpair1.13113xhardwareunchanged.
HistoricalvalidatedQwen4K QKV/output/FFNfirstcall1.2414/1.2429/1.2906x;second
1.2105/1.2247/1.2241x. Thosepre-retentioncomponentprofiles/FP32accumulation not
pooledwithcurrentFA; notwarmconvergence orend-to-endmeasurement.
Open:FA40four4Kcasesunaccepted/stale,packedproduceractualcorrectnessfailure,
4Kpacing/combinedsoftwareinteractionunmeasured,Llamatuningunfinished. No
upperboundorproofmajorarchitecturechangesnecessary. Existingvaliddata supports
modestsoftwareheadroom,notobviouslargeFAgainfrommoregenericSPMknobs.
OneCPU2GiBASanalysis; existingqueues/model/filesuntouchedexceptaudit/logdocs.

### FA-45 / FA-40 / FA-41 ordered continuation (2026-10-05)
User authorized finishing diagnostics, target software confirmation, bounded baseline
selection, then Llama/component consolidation. Host process audit found no live
FA40/41 simulators; old namespace PIDs were not host identities. All original
frozen inputs, accepted markers/output hashes verified; partial evidence hashed
and retained. Released only two orphaned 10GiB leases. Recovery receipt:
results/ordered_fa_followup_20261005/recovery_verified.json. FA40 imports34accepted,
restarts only4missing4K cases; numerical/config/resource validation body unchanged.
First continuation launch failed on directory-vs-status path before simulation;
evidence launch_failure_1, fixed path only. Detached pipeline now authoritative.
FA45 new isolated ELF adds bounded address specialization inside paced copy helper.
Six same-ELF arms cache/cache_stage/burst/stage/paced/combined, eachT256/T512 gates
thenT4096; no hardware/timing/geometry changes. Existing FA39/44 output references
reused, full oracle/config/work/copies/bounds retained. 1024 address encodings
checked equivalent, build PASS. New controls required by code-layout change;
not retuning previously established geometry. FullROI, not flitmeans, selects
bounded baseline. If any gate fails, pipeline stops before dependent phases.
Then FA41 continuation imports5accepted, same adaptive algorithm/validator.
Final scientific review still required, especially unresolved local frontier.
Historic projection/FFN profiles remain distinct, not integrated layer timings.
Shared20%CPU/RAM/64GiBheadroom; serialFA40, then6software sims/12allowedCPUs,
then1Llama. Each10GiBlease/8GiBinitialAS plus admittedgrowth. Source/guest/model
of previous studies unchanged. New log sources registered. No new runtime
acceptance or baseline freeze claimed at launch.

### FA-46 — Overnight operation-substitution screen (2026-10-05)
User authorized full-kernel packing/PV headroom tests to make an overnight
pursue/stop decision. Reuses FA39/SYN01 counter evidence and MOD11 scheduling;
not repeating standalone PV analytical sweep. New isolated guest/pseudo_inst
object+link; main binaries/objects and prior queues untouched. No performance
acceptance yet. Four host tests PASS; guest compile and isolated link PASS;
actual firstfour configs PASS. Root results/cacheflex_flash_attention/ncc_probe_20261005;
source benchmarks/qwen/fa/ncc_probe. Read README.txt for functional-proxy limits.
11arms: sameELF cache/SPM, extraDSBcontrol, Kpack/Ppack timed-memory proxies,
PVzero-transferoracle, PVexactoperand-loadprobe,64/128/256lane serializedbudget,
L3deliverybudget. Existing V-only layout/QK/softmax/FP32recurrence retained.
PV actualclaimedV + realpackedP, SoftFloat sequentialFP16FMA; no precomputedoutputs.
Host functional writes/read do NOT model real engine/coherence traffic; timed
operand probes plus service budgets are sensitivities, not realistic hardware
speedups or strict upper/lower bounds. L3case doesNOTrelocateV/modelNoC;
noenginequeue/arbitration/SRAMarea claim. Oneblockingcommand/core, DSB retained.
Lane costs5472/3424/2400cycles;L3budget6152, includes explicit serializedmemory
andcommandcosts. Fulloracle/exacthistoricalbits/config/work/copy/count/finitebounds
required; failures remain unaccepted. T256->T1024, max4workers/8allowedCPUs,
10GiBleaseeach/8GiBinitialAS+admittedgrowth, shared20%RAM/CPU+64GiBheadroom.
Hostcoordinator1351104; firstfour gates active. PipelineFA40/45/41 continues.
AutomaticfullROI threshold5% promotesatmost2nonoracle arms withmatchedcontrols
toT4096; nooracle-onlypromotion. Scientificreview needed evenoncompletion.
Sourcebuildattemptbeforecorrecthooksignaturepreserved guest_pre_hook_fix;
no simulation ran fromthat attempt. New log sources discover arms ascreated.

**FA-46 overnight screening finished:** accepted arms []; failed/unaccepted [{'arm': 'spm', 'code': 1, 'details': []}, {'arm': 'drained', 'code': 1, 'details': []}, {'arm': 'pv_oracle', 'code': 1, 'details': []}, {'arm': 'ppack', 'code': 1, 'details': []}, {'arm': 'kpack', 'code': 1, 'details': []}, {'arm': 'pv_traffic', 'code': 1, 'details': []}, {'arm': 'pv128', 'code': 1, 'details': []}, {'arm': 'pv64', 'code': 1, 'details': []}, {'arm': 'pv256', 'code': 1, 'details': []}, {'arm': 'pv_l3_budget', 'code': 1, 'details': []}, {'arm': 'cache', 'code': 1, 'details': []}]. Read ncc_probe_20261005_v4/REPORT.txt and target_decision.json. Functional proxy/serialized service sensitivities, not hardware performance claims.

**FA46 runtime gate repair and current queue:** Initial unsigned m5_sum interface
truncated guest pointers; failed readBlob attempts preserved in ncc_probe_20261005.
v2 explicit64bit alias fixed pointers but functional output writes required CPU
ordering; v3 postDSB passed both packing gates, PV still128incorrect elements.
v4 adds ISB pipeline serialization around diagnostic boundaries; allfour initial
T256 gates nowPASS fulloracle/exacthistoricalbits/config/counts/bounds. Root
results/cacheflex_flash_attention/ncc_probe_20261005_v4/status.json authoritative,
source benchmarks/qwen/fa/ncc_probe_v4. Existing isolatedbinary unchanged.
Guest linked with -Wl,-z,separate-code to provide safe exporter padding; original
v4 packagingfailure preserved prelink_failure (no simulations from that build).
Exact build.sh/compiler/source/ELF frozen. No tolerance/acceptance relaxation.
T256 sameELF cachepending, SPM134.1432us, drained134.6468, PVoracle119.3816,
Ppack133.8228. These are correctness gates, not target/realhardware results.
Allfour T1024 nowactive; sixremaininginterventions/cachepending, max4workers.
Hostcoordinator1702902. Failedv1/v2/v3partials/gates preserved, not pooled as
currentperformance; stopped only this diagnostic processgroups. FA40 remains
activeunchanged. ISB/DSB overhead means this is a conservative blocking proxy
in synchronization but optimistic functional transfers; no rigorousbound claim.
TimedPVoperand probes and finitebudgets stillmustpass their own gates.
Report should guide pursuit, not establish SPM-specific novelty by itself:
packing improvements could also benefit coherent execution. Any positivePV case
needs a real finite engine/operand interface and coherent-counterpart analysis.

**FA46 status review (2026-10-05):** Current v4 has9accepted records:
6T256 gates and3T1024 targets. T1024 ordinarySPM951.956us, addeddrain965.3968
(+1.412%latency), Kpack956.7504(+0.504%). Kpackproxy gives no demonstratedgain.
PVoracle andPVtraffic both passedT256 thenT1024 failed Ruby functionalread
(addresses0xab60c0/0xa9d180); Ppack passedT256 thenT1024 failed fulloracle
446/917504elements, maxabs9.53674e-5. Failures UNACCEPTED, no speedup claims
or architectural rejection from them; smallgate success didnotestablish
robust full-size functional transport. Fourfinitebudget T256 gates live,
cachepending; no T4096promotion. FA40targetcontrol also live, all5simsunpaused.
FA45softwareconfirmation andFA41Llama stillqueued behindFA40. Sharedreservation
56/150.987GiB, actualuserRSS6.37GiB, hostavailable596GiB, pressurefalse.
Currentlog367accepted/zerodiscrepancies. Near-cache pursue/stop decision still
unresolved pending valid targetPV transport/results; existingproxy needs repair.

**FA-40 recovery complete:** all38 accepted, original scientific validation unchanged. Target microseconds {"target_cache_control": 10432.5348, "target_spm_control": 9418.3056, "target_spm_bridge4096": 9428.1984, "target_spm_copy_oracle": 9266.826, "target_spm_combined": 9270.5968}. Oracles remain diagnostic only, not candidate baselines. Authoritative continuation results/ordered_fa_followup_20261005/fa40/status.json.

**FA-45 completed software confirmation:** 18 full-oracle cases accepted. Same-ELF bounded baseline cache cache_stage 10791.4328 us; SPM combined 9622.3884 us; ratio 1.1214921235147814x. See results/ordered_fa_followup_20261005/baseline_selection.json. Nominal retained hardware, fixed screened geometry; no global optimum/end-to-end claim.

**2026-10-05 overnight status audit:** FA40complete38accepted: copyoracle4K
1.608%lower thanSPMcontrol; bridge4096slightlyslower; combined1.568%lower.
FA45complete18accepted: combinedpaced+address9622.3884us vsburst9872.574
(2.534%lower), matchedbestcache10791.4328(1.12149x). NewELF absoluteSPM
is slower than earlier9418.3056/9225.5676 results; this is NOT a replacement
for the fastest historical valid kernel/globalfreeze. Keepmatchedprofile scopes.
LlamacacheBc12825845.124us vsBc25623499.3756 (9.982%slower); SPMBc128active.
FA46 fourfiniteT256 jobs alive but all8core instruction totals unchanged over
extendedsimulatedtime, zeroIPC; NOT healthyprogress. Preserved status/cmd/logtails
at ncc_probe_20261005_v4/stalled_finite_jobs.json and terminated ONLY verified
fourstalledsimulatorPIDs. Parentworkers recordfailures/releaseleases; independent
cachecontrol maythenstart. NoacceptedfinitePVresults, noarchitecturalconclusion.
Earlierproxyfunctionalread/numericalfailures remainunaccepted; implementation
needsrepair. Correcting priorheartbeat-only interpretation: freshstatus/loggrowth
wasnotproof ofguestprogress. Llama andorderedpipeline untouched.

**FA-46 overnight screening finished:** accepted arms ['drained', 'spm', 'kpack', 'cache']; failed/unaccepted [{'arm': 'pv_oracle', 'code': 1, 'details': [{'error': "AssertionError(': 35869, IPC: 01.43476\\n5910000000: Event_1609: system.cpu6 progress event, total committed:1265561, progress insts committed: 57325, IPC: 0002.293\\n5910000000: Event_1608: system.cpu5 progress event, total committed:1174785, progress insts committed: 29973, IPC: 01.19892\\n5910000000: Event_1607: system.cpu4 progress event, total committed:1270571, progress insts committed: 50373, IPC: 02.01492\\n5910000000: Event_1606: system.cpu3 progress event, total committed:1209303, progress insts committed: 56326, IPC: 02.25304\\n5910000000: Event_1605: system.cpu2 progress event, total committed:1194791, progress insts committed: 44598, IPC: 01.78392\\n5910000000: Event_1604: system.cpu1 progress event, total committed:1274052, progress insts committed: 53568, IPC: 02.14272\\n5910000000: Event_1603: system.cpu0 progress event, total committed:45976317, progress insts committed: 50321, IPC: 02.01284\\n5920000000: Event_1603: system.cpu0 progress event, total committed:46034450, progress insts committed: 58133, IPC: 02.32532\\n5920000000: Event_1604: system.cpu1 progress event, total committed:1285008, progress insts committed: 10956, IPC: 00.43824\\n5920000000: Event_1605: system.cpu2 progress event, total committed:1249474, progress insts committed: 54683, IPC: 02.18732\\n5920000000: Event_1606: system.cpu3 progress event, total committed:1249646, progress insts committed: 40343, IPC: 01.61372\\n5920000000: Event_1607: system.cpu4 progress event, total committed:1284560, progress insts committed: 13989, IPC: 00.55956\\n5920000000: Event_1608: system.cpu5 progress event, total committed:1227183, progress insts committed: 52398, IPC: 02.09592\\n5920000000: Event_1609: system.cpu6 progress event, total committed:1317999, progress insts committed: 52438, IPC: 02.09752\\n5920000000: Event_1610: system.cpu7 progress event, total committed:1214497, progress insts committed: 52988, IPC: 02.11952\\n5930000000: Event_1610: system.cpu7 progress event, total committed:1271940, progress insts committed: 57443, IPC: 02.29772\\n5930000000: Event_1609: system.cpu6 progress event, total committed:1378122, progress insts committed: 60123, IPC: 02.40492\\n5930000000: Event_1608: system.cpu5 progress event, total committed:1278941, progress insts committed: 51758, IPC: 02.07032\\n5930000000: Event_1607: system.cpu4 progress event, total committed:1303353, progress insts committed: 18793, IPC: 00.75172\\n5930000000: Event_1606: system.cpu3 progress event, total committed:1301434, progress insts committed: 51788, IPC: 02.07152\\n5930000000: Event_1605: system.cpu2 progress event, total committed:1300308, progress insts committed: 50834, IPC: 02.03336\\n5930000000: Event_1604: system.cpu1 progress event, total committed:1328071, progress insts committed: 43063, IPC: 01.72252\\n5930000000: Event_1603: system.cpu0 progress event, total committed:46085592, progress insts committed: 51142, IPC: 02.04568\\nsrc/mem/ruby/system/RubyPort.cc:472: fatal: Ruby functional read failed for address 0xab60c0\\nMemory Usage: 3185416 KBytes\\n')", 'traceback': 'Traceback (most recent call last):\n  File "/data2/rsappidi/cacheflex_mc/benchmarks/qwen/fa/ncc_probe_v4/run.py", line 125, in main\n    log=(path/\'run.log\').read_text();assert code==0 and \'because exiting with last active thread context\' in log,log[-3000:]\nAssertionError: : 35869, IPC: 01.43476\n5910000000: Event_1609: system.cpu6 progress event, total committed:1265561, progress insts committed: 57325, IPC: 0002.293\n5910000000: Event_1608: system.cpu5 progress event, total committed:1174785, progress insts committed: 29973, IPC: 01.19892\n5910000000: Event_1607: system.cpu4 progress event, total committed:1270571, progress insts committed: 50373, IPC: 02.01492\n5910000000: Event_1606: system.cpu3 progress event, total committed:1209303, progress insts committed: 56326, IPC: 02.25304\n5910000000: Event_1605: system.cpu2 progress event, total committed:1194791, progress insts committed: 44598, IPC: 01.78392\n5910000000: Event_1604: system.cpu1 progress event, total committed:1274052, progress insts committed: 53568, IPC: 02.14272\n5910000000: Event_1603: system.cpu0 progress event, total committed:45976317, progress insts committed: 50321, IPC: 02.01284\n5920000000: Event_1603: system.cpu0 progress event, total committed:46034450, progress insts committed: 58133, IPC: 02.32532\n5920000000: Event_1604: system.cpu1 progress event, total committed:1285008, progress insts committed: 10956, IPC: 00.43824\n5920000000: Event_1605: system.cpu2 progress event, total committed:1249474, progress insts committed: 54683, IPC: 02.18732\n5920000000: Event_1606: system.cpu3 progress event, total committed:1249646, progress insts committed: 40343, IPC: 01.61372\n5920000000: Event_1607: system.cpu4 progress event, total committed:1284560, progress insts committed: 13989, IPC: 00.55956\n5920000000: Event_1608: system.cpu5 progress event, total committed:1227183, progress insts committed: 52398, IPC: 02.09592\n5920000000: Event_1609: system.cpu6 progress event, total committed:1317999, progress insts committed: 52438, IPC: 02.09752\n5920000000: Event_1610: system.cpu7 progress event, total committed:1214497, progress insts committed: 52988, IPC: 02.11952\n5930000000: Event_1610: system.cpu7 progress event, total committed:1271940, progress insts committed: 57443, IPC: 02.29772\n5930000000: Event_1609: system.cpu6 progress event, total committed:1378122, progress insts committed: 60123, IPC: 02.40492\n5930000000: Event_1608: system.cpu5 progress event, total committed:1278941, progress insts committed: 51758, IPC: 02.07032\n5930000000: Event_1607: system.cpu4 progress event, total committed:1303353, progress insts committed: 18793, IPC: 00.75172\n5930000000: Event_1606: system.cpu3 progress event, total committed:1301434, progress insts committed: 51788, IPC: 02.07152\n5930000000: Event_1605: system.cpu2 progress event, total committed:1300308, progress insts committed: 50834, IPC: 02.03336\n5930000000: Event_1604: system.cpu1 progress event, total committed:1328071, progress insts committed: 43063, IPC: 01.72252\n5930000000: Event_1603: system.cpu0 progress event, total committed:46085592, progress insts committed: 51142, IPC: 02.04568\nsrc/mem/ruby/system/RubyPort.cc:472: fatal: Ruby functional read failed for address 0xab60c0\nMemory Usage: 3185416 KBytes\n\n'}]}, {'arm': 'ppack', 'code': 1, 'details': [{'error': "AssertionError({'elements': 917504, 'nonfinite': 0, 'errors': 446, 'max_abs_error': 9.5367431640625e-05, 'tolerance': '2e-6 + 1% abs(rounded reference)'})", 'traceback': 'Traceback (most recent call last):\n  File "/data2/rsappidi/cacheflex_mc/benchmarks/qwen/fa/ncc_probe_v4/run.py", line 141, in main\n    export_hash(path/\'output.fp16\',log);arithmetic=validate(path/\'output.fp16\',np.load(ref),c[\'t\'],14,2,True,dense=c[\'t\']<=256)\n  File "/data2/rsappidi/cacheflex_mc/scripts/qwen_l3_reference.py", line 57, in validate\n    assert row[\'nonfinite\']==row[\'errors\']==0,row\nAssertionError: {\'elements\': 917504, \'nonfinite\': 0, \'errors\': 446, \'max_abs_error\': 9.5367431640625e-05, \'tolerance\': \'2e-6 + 1% abs(rounded reference)\'}\n'}]}, {'arm': 'pv_traffic', 'code': 1, 'details': [{'error': "AssertionError('d: 59854, IPC: 02.39416\\n5850000000: Event_1609: system.cpu6 progress event, total committed:1110765, progress insts committed: 61195, IPC: 002.4478\\n5850000000: Event_1608: system.cpu5 progress event, total committed:1024300, progress insts committed: 61501, IPC: 02.46004\\n5850000000: Event_1607: system.cpu4 progress event, total committed:1116245, progress insts committed: 62356, IPC: 02.49424\\n5850000000: Event_1606: system.cpu3 progress event, total committed:1067140, progress insts committed: 51630, IPC: 002.0652\\n5850000000: Event_1605: system.cpu2 progress event, total committed:1038855, progress insts committed: 62003, IPC: 02.48012\\n5850000000: Event_1604: system.cpu1 progress event, total committed:1114219, progress insts committed: 61511, IPC: 02.46044\\n5850000000: Event_1603: system.cpu0 progress event, total committed:45793529, progress insts committed: 59141, IPC: 02.36564\\n5860000000: Event_1603: system.cpu0 progress event, total committed:45847464, progress insts committed: 53935, IPC: 002.1574\\n5860000000: Event_1604: system.cpu1 progress event, total committed:1176517, progress insts committed: 62298, IPC: 02.49192\\n5860000000: Event_1605: system.cpu2 progress event, total committed:1096589, progress insts committed: 57734, IPC: 02.30936\\n5860000000: Event_1606: system.cpu3 progress event, total committed:1125996, progress insts committed: 58856, IPC: 02.35424\\n5860000000: Event_1607: system.cpu4 progress event, total committed:1178085, progress insts committed: 61840, IPC: 002.4736\\n5860000000: Event_1608: system.cpu5 progress event, total committed:1077496, progress insts committed: 53196, IPC: 02.12784\\n5860000000: Event_1609: system.cpu6 progress event, total committed:1167333, progress insts committed: 56568, IPC: 02.26272\\n5860000000: Event_1610: system.cpu7 progress event, total committed:1071013, progress insts committed: 39511, IPC: 01.58044\\n5870000000: Event_1610: system.cpu7 progress event, total committed:1082131, progress insts committed: 11118, IPC: 00.44472\\n5870000000: Event_1609: system.cpu6 progress event, total committed:1229119, progress insts committed: 61786, IPC: 02.47144\\n5870000000: Event_1608: system.cpu5 progress event, total committed:1143371, progress insts committed: 65875, IPC: 0002.635\\n5870000000: Event_1607: system.cpu4 progress event, total committed:1234849, progress insts committed: 56764, IPC: 02.27056\\n5870000000: Event_1606: system.cpu3 progress event, total committed:1186646, progress insts committed: 60650, IPC: 0002.426\\n5870000000: Event_1605: system.cpu2 progress event, total committed:1163048, progress insts committed: 66459, IPC: 02.65836\\n5870000000: Event_1604: system.cpu1 progress event, total committed:1234733, progress insts committed: 58216, IPC: 02.32864\\n5870000000: Event_1603: system.cpu0 progress event, total committed:45857384, progress insts committed: 9920, IPC: 000.3968\\nsrc/mem/ruby/system/RubyPort.cc:472: fatal: Ruby functional read failed for address 0xa9d180\\nMemory Usage: 3179268 KBytes\\n')", 'traceback': 'Traceback (most recent call last):\n  File "/data2/rsappidi/cacheflex_mc/benchmarks/qwen/fa/ncc_probe_v4/run.py", line 125, in main\n    log=(path/\'run.log\').read_text();assert code==0 and \'because exiting with last active thread context\' in log,log[-3000:]\nAssertionError: d: 59854, IPC: 02.39416\n5850000000: Event_1609: system.cpu6 progress event, total committed:1110765, progress insts committed: 61195, IPC: 002.4478\n5850000000: Event_1608: system.cpu5 progress event, total committed:1024300, progress insts committed: 61501, IPC: 02.46004\n5850000000: Event_1607: system.cpu4 progress event, total committed:1116245, progress insts committed: 62356, IPC: 02.49424\n5850000000: Event_1606: system.cpu3 progress event, total committed:1067140, progress insts committed: 51630, IPC: 002.0652\n5850000000: Event_1605: system.cpu2 progress event, total committed:1038855, progress insts committed: 62003, IPC: 02.48012\n5850000000: Event_1604: system.cpu1 progress event, total committed:1114219, progress insts committed: 61511, IPC: 02.46044\n5850000000: Event_1603: system.cpu0 progress event, total committed:45793529, progress insts committed: 59141, IPC: 02.36564\n5860000000: Event_1603: system.cpu0 progress event, total committed:45847464, progress insts committed: 53935, IPC: 002.1574\n5860000000: Event_1604: system.cpu1 progress event, total committed:1176517, progress insts committed: 62298, IPC: 02.49192\n5860000000: Event_1605: system.cpu2 progress event, total committed:1096589, progress insts committed: 57734, IPC: 02.30936\n5860000000: Event_1606: system.cpu3 progress event, total committed:1125996, progress insts committed: 58856, IPC: 02.35424\n5860000000: Event_1607: system.cpu4 progress event, total committed:1178085, progress insts committed: 61840, IPC: 002.4736\n5860000000: Event_1608: system.cpu5 progress event, total committed:1077496, progress insts committed: 53196, IPC: 02.12784\n5860000000: Event_1609: system.cpu6 progress event, total committed:1167333, progress insts committed: 56568, IPC: 02.26272\n5860000000: Event_1610: system.cpu7 progress event, total committed:1071013, progress insts committed: 39511, IPC: 01.58044\n5870000000: Event_1610: system.cpu7 progress event, total committed:1082131, progress insts committed: 11118, IPC: 00.44472\n5870000000: Event_1609: system.cpu6 progress event, total committed:1229119, progress insts committed: 61786, IPC: 02.47144\n5870000000: Event_1608: system.cpu5 progress event, total committed:1143371, progress insts committed: 65875, IPC: 0002.635\n5870000000: Event_1607: system.cpu4 progress event, total committed:1234849, progress insts committed: 56764, IPC: 02.27056\n5870000000: Event_1606: system.cpu3 progress event, total committed:1186646, progress insts committed: 60650, IPC: 0002.426\n5870000000: Event_1605: system.cpu2 progress event, total committed:1163048, progress insts committed: 66459, IPC: 02.65836\n5870000000: Event_1604: system.cpu1 progress event, total committed:1234733, progress insts committed: 58216, IPC: 02.32864\n5870000000: Event_1603: system.cpu0 progress event, total committed:45857384, progress insts committed: 9920, IPC: 000.3968\nsrc/mem/ruby/system/RubyPort.cc:472: fatal: Ruby functional read failed for address 0xa9d180\nMemory Usage: 3179268 KBytes\n\n'}]}, {'arm': 'pv128', 'code': 1, 'details': [{'error': "AssertionError('cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2375070000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375080000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2375090000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n')", 'traceback': 'Traceback (most recent call last):\n  File "/data2/rsappidi/cacheflex_mc/benchmarks/qwen/fa/ncc_probe_v4/run.py", line 125, in main\n    log=(path/\'run.log\').read_text();assert code==0 and \'because exiting with last active thread context\' in log,log[-3000:]\nAssertionError: cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2375070000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375080000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2375090000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n\n'}]}, {'arm': 'pv64', 'code': 1, 'details': [{'error': "AssertionError('cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2359610000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359620000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2359630000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n')", 'traceback': 'Traceback (most recent call last):\n  File "/data2/rsappidi/cacheflex_mc/benchmarks/qwen/fa/ncc_probe_v4/run.py", line 125, in main\n    log=(path/\'run.log\').read_text();assert code==0 and \'because exiting with last active thread context\' in log,log[-3000:]\nAssertionError: cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2359610000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359620000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2359630000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n\n'}]}, {'arm': 'pv256', 'code': 1, 'details': [{'error': "AssertionError('cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2343670000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343680000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2343690000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n')", 'traceback': 'Traceback (most recent call last):\n  File "/data2/rsappidi/cacheflex_mc/benchmarks/qwen/fa/ncc_probe_v4/run.py", line 125, in main\n    log=(path/\'run.log\').read_text();assert code==0 and \'because exiting with last active thread context\' in log,log[-3000:]\nAssertionError: cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2343670000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343680000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2343690000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n\n'}]}, {'arm': 'pv_l3_budget', 'code': 1, 'details': [{'error': "AssertionError('cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2324330000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324340000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\\n2324350000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\\n')", 'traceback': 'Traceback (most recent call last):\n  File "/data2/rsappidi/cacheflex_mc/benchmarks/qwen/fa/ncc_probe_v4/run.py", line 125, in main\n    log=(path/\'run.log\').read_text();assert code==0 and \'because exiting with last active thread context\' in log,log[-3000:]\nAssertionError: cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2324330000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324340000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1610: system.cpu7 progress event, total committed:31422, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1609: system.cpu6 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1608: system.cpu5 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1607: system.cpu4 progress event, total committed:31412, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1606: system.cpu3 progress event, total committed:41320, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1605: system.cpu2 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1604: system.cpu1 progress event, total committed:41310, progress insts committed: 0, IPC: 00000000\n2324350000000: Event_1603: system.cpu0 progress event, total committed:11285795, progress insts committed: 0, IPC: 00000000\n\n'}]}]. Read ncc_probe_20261005_v4/REPORT.txt and target_decision.json. Functional proxy/serialized service sensitivities, not hardware performance claims.

**FA46 v5 monitored repair started (2026-10-05):** Explicit user request repair
and monitor throughT1024. V4finished withfourfullarms(cache/SPM/drained/Kpack),
Ppacknumerical/PVfunctionalread failures andfourfinitejobsstalled; allpreserved.
New isolated guest+hook+link at ncc_probe_20261005_v5; source correspondingv5.
Coherent payload nowactual CPU loads/stores viafour256B SVE registers perchunk,
no Rubyfunctional coherent read/write. PrivateVhookread unchanged. Outputregister
writes followedISB, completechunkordering/bounds/bytecounts mandatory.
ReadyTick+CPU polling replaces quiesceCycles; retains explicitservicebudgets,
no O3suspension. Oneblockingcommand/core; no realengine/NoCtiming claim.
This changes diagnostictransportoverhead; same-newELF cache/SPMcontrols required.
FivehosttestsPASS, guest/isolatedcompile/linkPASS. FirstfourT256gates started
(spm,pv128,pvoracle,ppack), runtimeacceptancepending. Progresswatchdog120hostsec
withoutcommittedinstructionadvance;30minsmall/90mintargetlimit,pressurepauses
excluded fromnoprogress. Max4workers,shared20%CPU/RAM+64GiBheadroom unchanged.
No T4096 autopromotion inthisrepair; userrequestedT1024completionfirst. Llama
untouched. Newcurrentlogsourcev5; oldresultsnotpooled. Coordinator835495.

**FA46v5 firsttargetacceptance:** FourT1024 cases complete with917504elements,
zerooracleerrors, exacthistoricalpeerbits andtransport/service/count/config gates.
SameELF SPM933.238us, Ppack1034.2472, PVno-added-service850.4608,
PV128budget1438.6504. Formerfunctionalread/numerical andsuspension failures not
reproduced withregistertransport/polling. No-added-service PV8.869%lower runtime;
this includes timedP/result registertransport butfunctionalV/noenginearithmetic
service. PV12854.16%slower, blockingserializedbudget only, notrealengine latency.
Remainingarmsstillrunning; nohardware feasibility/rejection orT4096claim.

**FA-46 overnight screening finished:** accepted arms ['pv_oracle', 'spm', 'ppack', 'pv128', 'pv_traffic', 'pv256', 'pv64', 'pv_l3_budget', 'kpack', 'drained', 'cache']; failed/unaccepted []. Read ncc_probe_20261005_v5/REPORT.txt and target_decision.json. Register-transport proxy/serialized service sensitivities, not hardware performance claims.

**FA46v5 MONITORED COMPLETE (2026-10-05):**22/22accepted(11T256+11T1024),
zero runtime/numerical/config/transport/count/watchdog failures. All917504target
outputs exacthistoricalbits; oneguest/onebinary, independentoraclezeroerrors.
Final rawoutputhash/markerreceipts at ncc_probe_20261005_v5/FINAL_VERIFICATION.json;
REPORT.txt conciseacceptedtable. AllFA46leasesreleased; noT4096launched; Llama
untouched. T1024us: cache996.804,SPM933.238,drained940.7168,Kpack933.8352,
Ppack1034.2472,PVno-service850.4608,PVtraffic956.0532,PV64=1723.4624,
PV128=1438.6504,PV256=1294.4568,L3deliverybudget1819.3292.
PVno-service8.870%lower; retainedoperandprobe2.445%higher; finitebest256lanes
38.706%higher. Kpack+0.064%effectivelynull; Ppack+10.824%. Cache/SPM1.06811x
withinthisguest only. No-added-service test includesnewCPUregisterP/result
transport butfunctionalV/noarithmeticservice. Finitebudgets blocking/additive,
polling/DSB/ISB/adaptercosts retained; PVtraffic duplicates somePdelivery.
Notactualengine/NoCspeedups, notrigorousbounds, noasynchronousdesign rejection.
Repair removesbulkfunctional coherentmemory andquiesce wake dependencies, NOT
changes to ordinaryCacheFlexcoherence/CPU/memorytiming. Priorfailedattempts
preserved andnotaccepted. User's request to monitorthroughT1024 isfulfilled.

**FA-47 asynchronous PV implemented / T1024 validation starting (2026-10-05):**
User explicitly requested asynchronous PV. Builds on accepted FA-46v5 transport,
not a repeat of its blocking experiment. Isolated guest/hook compile/link PASS;
four independent recurrence/lifetime/accounting tests PASS, including negative
alpha-reuse/V-restaging controls and 100 schedule seeds. Depth-one per-core PV
retains alpha and destination while CPU computes next row QK/softmax/P packing;
drain before command-buffer reuse, V restaging and normalization. Matched blocking
128/256 budgets and cache/SPM controls in same guest. Six T256 gates then their
T1024 targets; full oracle/exact bits/work/copies/transport/config/LSQ checks plus
measured useful CPU/service overlap required. No T4096 promotion. Root
results/cacheflex_flash_attention/pv_async_20261005; source benchmarks/qwen/fa/pv_async.
Max4simulators via shared20%broker, initial10GiB leases,2CPUs each,64GiB extra
headroom; Llama unchanged. Existing FA46 completed, no duplicate live PV queue.
Decision: compare full ROI async/blocking and ordinary SPM, with actual positive
overlap required. Functional V/additive service/no bank contention and duplicated
P traffic caveats remain; results cannot establish hardware speedups or an upper
bound. No accepted asynchronous runtime at launch.

**FA-47 small gates accepted:** all four engine T256 cases pass full independent
oracle, historical exact output bits, exact work/transport/service, config/bounds
and overlap validation. Blocking128/async128=183.9584/143.8312us;
blocking256/async256=170.7460/143.1620us. Useful-work/service intersection
285400400/199384400 ticks in async128/256 versus zero in both blocking controls.
These are validation-size results only; all four T1024 targets now active.

**FA-47 T1024 asynchronous correctness/overlap verified:** all four engine arms
accepted full oracle and historical exact bits for917504outputs,2240PVcommands,
copy/transport/config/LSQ bounds, V lifetime and completion checks. Blocking128
1499.9744us -> async1281020.9100us; blocking2561355.6936us -> async2561009.5000us.
Useful CPU/service overlap2854050800/2000304800ticks versuszero inblocking.
Blocked ready polls60849/41363versus956472/669752. Ordinary same-guest cache/SPM
controls still running; do not claim net gain over SPM until accepted. This
verifies asynchronous scheduling in the diagnostic ready-time model, not a
finite-resource hardware implementation. All previous FA-46 caveats retained.

**FA-47 overnight screening finished:** accepted arms ['async128', 'async256', 'block256', 'block128', 'spm', 'cache']; failed/unaccepted []. Read pv_async_20261005/REPORT.txt and analysis.json. Asynchronous register-transport/service-time proxies, not hardware performance claims.

**FA-47 COMPLETE — asynchronous PV verified at T1024 (2026-10-05):**
All 12/12 cases accepted (six T256 gates and six T1024 targets), no failures.
Final cross-arm audit verified 2,257 frozen-file hashes and all receipt/output
hashes. T1024: 917,504 outputs match historical controls and all peers bit-for-bit;
full independent oracle, 2,240 PV commands, exact transport/copy/work counts,
hardware configuration, LSQ bounds, completion and V-buffer lifetime checks pass.
Same-guest full ROI (us): cache 1005.0604; ordinary SPM 957.6336;
blocking128 1499.9744 / async128 1020.9100;
blocking256 1355.6936 / async256 1009.5000.
Async cuts blocking runtime 31.938% / 25.536%; useful QK/softmax/P-packing regions
overlap 93.029% / 93.020% of modeled engine service. Blocking overlap is zero.
Unsuccessful completion polls fall 956472->60849 / 669752->41363.
Net result remains NEGATIVE versus ordinary SPM: async128 6.608% slower,
async256 5.416% slower. Therefore asynchronous scheduling works, and rejecting PV
on blocking results alone was unwarranted, but this implementation has not shown
a net PV benefit. Do not attribute the remaining gap to a single cause without
another matched diagnostic. Service overlap summed across cores is not runtime
saved. Model retains functional V reads, additive service budgets, register
payload transport/DSB/ISB/polling and duplicated P operand probes; no bank/NoC
contention or hardware feasibility claim. No T4096 runs launched. All FA-47 leases
released; unrelated Llama queue preserved. Results and final audit:
results/cacheflex_flash_attention/pv_async_20261005/{REPORT.txt,FINAL_VERIFICATION.json}.

**FA-48 asynchronous PV attribution diagnostics launched (2026-10-05):**
User authorized diagnostics to pinpoint FA-47's residual net slowdown. FA-46's
blocking sensitivity and FA-47's overlap/poll counts cannot distinguish common
CPU work slowdown from adapter/transfer/tail costs. New source pv_diag and root
results/cacheflex_flash_attention/pv_diag_20261005, isolated guest/hook/observer
build PASS; five host accounting/plan/map tests PASS. Passive commit-PC phase
observer first tested OFF/ON on same-guest ordinary SPM and async256 at T256 and
T1024; all non-host stats/config/output must be exact before interventions launch.
Then five matched interventions: remove timed CPU P/V operand probe; zero engine
service; both; remove two redundant outer fence pairs; probe+outer-fence removal.
Nine arms,18 total gate/target cases, no4K. All original oracle/exact math/work/
copy/transport/config/LSQ/lifetime checks retained plus phase partition and counts,
zero-service zero-blocked-polls and removed-probe zeroCPU-SPM-load-response checks.
Expected decision: full ROI factorial effects plus per-core phase timelines show
whether extra operand work, tail service, transport or common QK/softmax dilation
explains the gap. Do not sum phase ticks across cores as runtime or add causal
intervention effects. New labels may alter compiler layout; only same-guest pairs
are causal. Functional V, additive service and missing bank/NoC contention remain
limitations. All old results preserved; Llama unchanged. Max5workers,2CPUs and
10GiB initiallease each; shared20%CPU/RAM,64GiB extra headroom, progress watchdog.
No accepted FA48 runtime at launch.

**FA-48 observer validated at both sizes; interventions active:** eight controls
accepted. T1024 observer OFF/ON exactly identical all non-host stats and output
bits: ordinary SPM945.2608us,async990.7576us (new guest,4.813% slower). Every
per-core timeline partitions ROI and matches query-block counts. Average per-core
QK164.7271->166.2971us,softmax255.5299->257.5239us; modest +3.5638us combined.
CPU PV+recurrence165.4672us replaced by probe86.8980,input chunks45.6862,result
transfer19.7694,recurrence23.4132,completion wait18.5614us plus smaller overheads.
These are phase occupancies, NOT summed wall-runtime attribution. Five removal
interventions now active; causal attribution awaits their accepted full ROI.

**FA-48 overnight screening finished:** accepted arms ['spm_off', 'async_off', 'spm', 'async', 'no_probe', 'no_probe_zero', 'no_probe_less_fence', 'zero_service', 'less_fence']; failed/unaccepted []. Read pv_diag_20261005/REPORT.txt and analysis.json. Asynchronous register-transport/service-time proxies, not hardware performance claims.

**FA-48 first18 COMPLETE, attribution caveat / inert controls required:** all18
full validations pass;2,277 frozen hashes and output files audited. New-guest
T1024us:SPM945.2608,async990.7576,no_probe923.7396,zero_service1010.7436,
no_probe_zero902.8596,less_fence1022.3172,no_probe_less_fence913.4156.
HOWEVER flags1..5 have EXACTLY identical absolute ROI start5254541200 and all
8 K-packing phase durations; flag0 async starts5255128000 and differs in K pack,
BEFORE intervention. Thus raw effects versus flag0 are confounded by startup
state; do not call67.018us exclusively a probe-removal causal effect, nor claim
zero service intrinsically hurts. Nonzero-flag pair comparisons remain informative.
New continuation source benchmarks/qwen/fa/pv_diag_control reuses exact guest,
binary,source/model; guest flag2 retains probe/fences, engine env flag0 restores
2400cycle service. Inert async plus ordinary spm_inert,2T256+2T1024 cases. Inert
command args match zero_service EXACTLY; only engine env differs. Require matching
startup/K-pack timelines for causal interpretation. Existing18 results preserved.
Baseline nonzero-flag control also measures startup sensitivity; no hardware or
precision changes.2workers10GiB/2CPUs each under same20%/64GiB constraints.

**FA-48 FINAL COMPLETE — dominant async PV overhead identified (2026-10-05):**
22/22 accepted including four inert-startup cases. Full917504output oracle/exact
bits,2240PVcommands,71680copies,transport/service/lifetime/config/LSQ and phase
accounting checks pass;2,283frozen hashes and all outputs audited. Passive observer
ON/OFF exact all non-host statistics. Inert engine control matches absolute ROI
start and phase0/1/2 (including entries) on ALL8cores with EVERY intervention at
T256 andT1024. Earlier flag0 causal-effect interpretations are superseded.

Startup-matched T1024us: ordinary spm_inert950.7684; finite async inert1029.9156;
no_probe923.7396;zero_service1010.7436;no_probe_zero902.8596;
less_fence1022.3172;no_probe_less_fence913.4156. Probe removal saves106.1760us
(10.309%) versus inert; zero-service19.1720us(1.862%); outer fences7.5984us
(0.738%). After removing probe, zero-service saves20.8800us and fewer outer
fences10.3240us. Thus retained CPU P/V read loop is the DOMINANT demonstrated
avoidable overhead, not a failure of asynchronous execution. It was carried over
from traffic-preserving diagnostics despite real register P/result transport and
engine operand-service budgets. Probe removal eliminates573479CPU-SPM responses
(inert) and~1.736million committed loads (4839665->3103662), no math/copy changes.

Finite no-probe PV is2.843% lower latency than matched ordinary SPM; removing
outer fences too yields3.929% lower (1.040894x). This is diagnostic-model headroom,
not a finite-resource hardware speedup. Common QK+softmax average per core rises
420.8487->430.2561us(+2.235%) versus ordinary SPM: modest dilation, not the main
extra cost. Remaining average/core input chunks45.9532us,result transfer21.6935,
recurrence23.6877,completion wait18.5701; phase occupancy is NOT additive wall-time
attribution or independent speedup potential. Functional V/additive service/no
bank-NoC arbitration remain explicit. No4K or new hardware claim. Initial18
receipts retained; underlying startup-state cause not uniquely established, but
matched inert controls eliminate its confounding of these intervention effects.
Read pv_diag_20261005/REPORT.txt,DIAGNOSIS.json,FINAL_VERIFICATION.json; source
pv_diag plus pv_diag_control. All diagnostic jobs complete/leases released,
20%CPU/RAM/64GiB headroom retained; unrelated Llama preserved.

**FA-49 per-core phase/overlap profiler implemented and launching (2026-10-05):**
User requested cache/SPM phase ROIs to guide architectural additions. FA-48 already
provided passive phase timing but no matched cache profile or copy/load/RAW
lifetimes by phase. New source benchmarks/qwen/fa/phase_profile; root
results/cacheflex_flash_attention/phase_profile_20261005. EXACT FA48 guest reused,
NCCmode0,flag2; ordinary cache/private V-only R256Br16Bc256,14Q/2KV,d64,8cores.
No new guest/layout, engine or ideal timing. Isolated pseudo_inst/Sequencer/LSQ
objects compile/link PASS; six independent interval/accounting/plan tests PASS.
Four OFF/ON arms, eachT256 thenT1024. Full prior oracle/exact bits/config/work/
copy/LSQ checks plus every non-host stat OFF/ON exact. SPM also must reproduce
FA48spm_inert exact stats. Do not call this new tuning or a fastest baseline.

New per-core phase records partition workbegin->workend; fused PV/recurrence
remains fused. Track Sequencer-copy issue/completion (not earlier CPU queue time),
physical cache/SPM load packets, RAW waiting loads including squash, and port
rejects. Phase union and count-time integrals keep overlap explicit. Event phase
is current retirement phase, not necessarily originating instruction. Requests
admitted before ROI excluded; censored memory packets at end counted. Pending
request/RAW-load intervals do not imply whole-core stalls. Full accounting and
legacy counter reconciliation required. Decision: compare phase concentrations,
critical-core completion and copy/consumer overlap to prioritize matched causal
architectural tests; no sum of multicore durations as wall-time contributions.
Max4workers,2allowedCPUs/10GiB initialleases, shared20%CPU/RAM and64GiB headroom.
Llama unchanged. No4K auto-promotion or accepted new runtime at launch.

**FA-49 small controls accepted / T1024 verification progressing:** fourT256
controls pass all checks and exact OFF/ON non-host stats/output. Cache131.5160us,
SPM136.3580us; SPM exactly matches historicalFA48spm_inert. New SPM records account
for7168issued/completedcopies and zero blocked RAW loads, reconcile all legacy
counters and partition phase time. Pending coherent packets at ROI end explicitly
reported (SPM cores3/5:5/3packets); no uncompleted copies/SPM loads/RAW waits.
This is expected censored physical traffic, not a validation failure. Target
SPM_OFF has also completed with historical exact equivalence; remaining target
ON/OFF checks ongoing. No target phase conclusion accepted yet.

**FA-49 overnight screening finished:** accepted arms ['spm_off', 'spm', 'cache_off', 'cache']; failed/unaccepted []. Read phase_profile_20261005/REPORT.txt and analysis.json. Passive phase and in-flight request observations, not hardware performance claims.

**FA-49 COMPLETE — cache/SPM phase and overlap profile validated (2026-10-05):**
8/8 accepted (OFF/ON cache/SPM atT256,T1024), zero failures. Every non-host stat
and output bit exactly OFF/ON; SPM matches acceptedFA48spm_inert exactly. Independent
full oracle,917504target outputs, work/copy/config/LSQ checks pass; phase partitions,
packet conservation, copy integral/latency and legacy RAW/port counters reconcile.
Final audit2,292frozen file hashes plus all output hashes PASS. No kernel guest,
coherence, scheduling or timing change; new isolated passive observer only.

T1024 matched diagnostic guest: cache1000.4952us,SPM950.7684us (1.052302x).
Last-arriving core6 in both. Critical-core cache->SPM phase us:
K/Qpack72.9292->71.9248; Vstage/setup0.1728->28.9284;
QK237.5692->198.2004;softmax322.1452->316.0212;
Ppack62.5480->53.7524;PV+recurrence238.1356->213.1364;
normalize54.9912->56.4828. Observational phase deltas locate gains, not separate
causal/removable costs. SPM critical-core shares of outerROI:softmax33.24%,
PV+recurrence22.42%,QK20.85%,staging3.04%. Profile guides matched compute/fusion
experiments before assuming direct-copy latency dominates; does not alone select
an architecture or establish headroom. PriorFA40copy-oracle was a causal test.

SPM71680copies issued/completed; zeroRAW-blocked loads/wait;4SPM-load port-block
attempts. Average/core copy-in-flight union24.2517us:22.8681duringVstage/setup,
1.3837duringQK;5.71%outside staging. NOT a hidden-latency fraction or whole-core
stall time. No copy/SPM/RAW outstanding at end; alltarget tracked coherent data
packets also complete. Cache/SPM physical coherent-load packet counts6943327/
5811573;SPM-load packets575696. Includes speculation, not committed load counts.
Per-core barrier time exposes T1024 coarse-work imbalance (up to~34%ROI on some
cores versus~0.2% on core6). R256 inherited,4querygroups/head and56tasks; this is
not a re-tuned T1024 or new best-baseline result, nor proof the same imbalance holds
at4K. Do not sum core times or pending-request intervals as wall contributions.

Artifacts at phase_profile_20261005: REPORT.txt,phase_analysis.json,
phase_counters.csv,phase_breakdown.{png,svg,pdf},FINAL_VERIFICATION.json.
One outerROI retained; phase timings measured at commit boundaries. Copy timing
begins at Sequencer issue, not earlier CPU LSQ residence. Event-phase labels may
differ from instruction origin under OOO. Runtime RAW is zero here; positive/
squashed interval arithmetic separately host-tested. AllFA49 jobs/leases released;
20%CPU/RAM/64GiB headroom maintained; unrelated Llama remains unchanged. No4K
promotion or hardware performance prediction.

### SYN-02 — Post-hoc memory-subsystem energy, multicore Qwen FA (2026-10-05)
User-requested analysis only; no simulations. Applies original paper RTL per-access
energies/idle powers (cacheflex_micro cacheflex_rtl_eval.xlsx, compute_energy.py)
unchanged to Ruby ROI counters of accepted FA-39 (T1024/T4096, 8/16 L1D credits)
and FA-35/40 matched pairs plus all accepted T4096 Hq14 cache/SPM matrix tilings.
Matched pairs, identical outputs: SPM total energy -21.4% (8 credits) / -17.6%
(16 credits, 1.015x) at T4096; -17.8%/-14.4% at T1024; on-chip -26 to -30%.
Saving is mainly private-L2 reads 44.4M->18.3M (L1 refill from V/K tile thrash).
Fair-baseline check: lowest-energy cache tiling Br16/Bc128/R512 192.6mJ vs fastest
213.1mJ; SPM speed winner 173.5mJ is 9.9% lower (best-vs-best 13.4%).
Energy-optimal cache tiling at 16 credits untested. Not modeled: NoC/coherence, TLB,
SPM LSQ, core/copy-instruction energy; 12nm single-core constants per Ruby request.
Estimate under paper model, not calibrated multicore energy.
[Report](../../results/cacheflex_energy_audit_20261005/REPORT.md).

### AI-13 — Qwen FFN/projections at 16 L1D miss credits (2026-10-05)
User requested comprehensive GEMM/FFN results accounting for higher MSHR counts.
Builds on AI-10/AI-11 (8 credits; cache winners at search edge MP8/KC512, credits
saturated: T4096 FFN 22.4M vs 1.87M L1D misses), FA-39 (8->16 credits shrinks FA
SPM lead 1.110x->1.015x), SYN-02 (fastest != lowest-energy tiling). Isolated source
benchmarks/qwen/components/mshr16; root results/ai_workloads/qwen_mshr16_20261005.
Frozen AI-10/11 build_v5, config script, guests, inputs; only override
l0_cntrl*.dataMissCredits.capacity=16 (TBEs16, targets16, L2 credits20, VL2048 kept).
Exact replay gate: two AI-11 T1024 runs at 8 credits must reproduce command, timings,
bits. Bounded MP8 KC{128,256,512,1024} both backends (B panel 32-256KiB vs
accumulator traffic tradeoff, analytical); T24 gates, T1024 screen, 8 cache-only
8-credit attribution cases; per-backend first/repeat-call winners (<=2) to T2048,
T4096. Unchanged AI-10/11 acceptance plus uniform-credit config audit, maxLive bound,
bits equal to accepted AI-10/11 outputs. Five host tests PASS incl. full validation
re-accepting old runs with identical timings/bits and config negatives. Max4 sims via
broker 10GiB leases, 8GiB AS+growth, budget CPU offsets4-11, 20%CPU/RAM, 64GiB
headroom; FA-41 untouched. Energy via SYN-02 model. No accepted AI-13 result at launch;
no global-optimum claim. Decision: repeat-call best-vs-best >=~1.10x credible GEMM
speedup; ~1.0-1.05x shifts claim to energy.

### OPS-24 — Home quota recovery and ring-mechanism archive (2026-10-05)
User-requested cleanup. Home (/homes, 5120M hard limit, was at limit with grace
running) reduced 5.0G->3.6G: removed unused Codex release 0.159.3 and a stalled
0.160.1 staging dir (installer dead), cleared ccache, npm and pip caches. Kept
~/.local (experiment numpy), Codex sessions, toolchains. /data2 has no quota.
Ring mechanism (FA-34 optimistic_ring_20261001, FA-38 ring_async_20261003, sources
benchmarks/qwen/fa/{ring,ring_async}) archived to archive/ring_mechanism_20261005
(tar.gz + per-file SHA256 manifests; every member re-hashed from the archive before
unlink). Kept in place: status.json/accepted.json/.md markers (ledger links and
current-results log valid: 497 accepted, 0 discrepancies) and all FA-37 dsb_diag
frozen ring dependencies (optimistic_ring build_retained_v1 incl. gem5.opt; 4 files
first removed then restored from archive after a key-based manifest check; all 79
FA-37 ring hashes verified). Not archived: v_only_ring_20261003 (FA-36; FA-37 guest
and control runs) and qwen_qo_ring_20260930 (FA-33 Q/O residency, not ring work).
Restore: tar -xzf archive/ring_mechanism_20261005/<tree>.tar.gz from repo root.

### FA-50 — Softmax/PV software optimizations, exact-bit first (2026-10-05)
User authorized implementing and verifying untried softmax software options after
FA-49 (softmax 33% of critical-core ROI, low memory activity) and the AI-13 FP
utilization check (~9% of FMA peak). Not tried before: LEG-01 two-row interleave was
an old invalid-kernel clue; FA-39 packed-P producer (failed) and L3-04/05 epilogues are
distinct. One ELF from accepted FA-39 guest_v1, runtime flags: 1 four-row softmax,
2 unit-alpha PV variant, 4 fmaxv QK-epilogue row max, 8 mask folded into pass 1,
16 Q pre-scale. Flag 0 = verbatim FA-39 softmax path. Flags 0/1/2/4/8/15 must equal
FA-39 output bits (T256 554df33e.., T1024 5f479413..); 16/31 full oracle, bit equality
recorded. Disassembly checked: noalpha PV variants 0 fmul/ld1rw (16 each in originals),
16-row fmaxv epilogue. Four host tests PASS (plan, FA-39 command equivalence, config
negative, verbatim flag-0 path). 16 T256 gates then T1024 cache/SPM flags0/15/31 gated
on both gates. Retained simulator/config unchanged; four lanes, broker 10GiB leases,
20%CPU/RAM, 64GiB headroom; AI-13/FA-41 untouched. Benefits are shared FA software
headroom, not SPM-specific evidence. No accepted FA-50 result at launch.

**FA-50 queue finished (complete_requires_review):** 22 accepted, 0 failed/skipped. Read results/cacheflex_flash_attention/softmax_opt_20261005/REPORT.md; scientific review required.

**AI-13b tuning-gap follow-up (2026-10-05):** AI-13 transfers only T1024 winners to
T2048/T4096; accepted data show the cache optimum moves with T (QKV T2048: AI-11
8-credit KC512 4768.6us repeat beats AI-13 16-credit KC1024 4892.8us). Also non-monotonic
credits: output T2048 cache KC512 16 credits 3646.1/3461.2us vs 8 credits 3631.5/3416.0us
(slower with more MSHRs; FA-23 also saw 16->32 regress). Separate queue
benchmarks/qwen/components/mshr16_neighbors (AI-13 sources untouched, imported):
after AI-13 finishes a (workload,backend,T), runs KC neighbors of the current first-
and repeat-call best until both neighbors are tested within KC{128..1024}; identical
AI-13 commands/acceptance (verified equal except outdir). Initial needs: QKV cache/SPM
KC512 and output cache KC256 at T2048. Slots offsets 12/14, broker leases, same
watchdogs. Merged analysis reports per-cell tiling coverage/convergence. Note: AI-11
8-credit output speedup is non-monotonic in T (1.154x/1.043x/1.225x repeat at
1K/2K/4K); single-tiling cells, unexplained, not interpreted.

### FA-51 — FA-50 software: T4096 attribution and Bc/R retuning (2026-10-05)
User authorized the tuning sequence after FA-50 (exact bits; T1024 flag15 -6.53%
cache / -7.09% SPM at inherited Br16/Bc256/R256; T256 overstated mask folding since every
block is a boundary block). Guest v2 = FA-50 build minus the FA-39 geometry lock; driver
diff 3 lines (row-max buffer sized for Bc/128 chunks, guard lifted, flag4 requires Br16).
Stages: (1) T4096 Bc256/R256 8-credit flags 0/1/2/4/8/15 both backends; (2) per backend
x credits{8,16} T1024 coordinate descent over Bc{128,256,512} x R{128,256,512}, Br16,
flag 15, 0.5% tie rule; (3) T4096 confirmation of winner, runner-up and winner flag 0.
Exact bits per (T,Bc) (bits independent of R/Br/flags/backend/credits, verified on FA-35
records): FA-35 references or first accepted peer; T1024 Bc128/512 flag-0 controls first.
Generalized work() reproduces 11 FA-35 per-core records; commands equal FA-50 except guest
path; config negatives. Slots offsets 16/18/20, broker leases, 900s watchdog, 20% CPU/RAM,
64GiB headroom. Shared software headroom, not SPM-specific evidence; bounded search.

**FA-51 finished (complete_requires_review):** 54 accepted, 0 failed. Read results/cacheflex_flash_attention/softmax_tune_20261005/REPORT.md; scientific review required.

**FA-51 review / FA-51b launch (2026-10-06):** FA-51 complete, 54/54 accepted, exact bits
throughout. Design flaw: T1024 descent moved all four cells to R128 (T1024 balance), but
at T4096 R256 is faster where measured (8 credits flag15: cache 10164.58 vs 10687.43 us;
SPM 8952.13 vs 9499.58 us); 16-credit Bc256/R256 at T4096 was never run, and FA-39's
16-credit R256 cache (8744 us, other guest) beats FA-51's R128 confirmation (8856.91 us).
Flag 2 alone slows T4096 (cache +0.84%, SPM +1.49%) though flag 15 combined is best seen.
FA-51b (followup.py, reuses Tuner.run/acceptance): T4096 Bc256/R256 16 credits f0/f13/f15
and 8 credits f13, both backends (8 runs). No conclusion from T1024-only geometry choices.

**FA-41 final review (2026-10-06):** 24/24 accepted re-verified (ROI and output hashes
re-hashed, 230 frozen inputs no drift, coordinator exited). T4096 Llama-shape local
winner Br16/Bc256/R256 for BOTH backends; all five one-step neighbors measured, none
better (cache Br32 +0.66% near-tie; others +3-16%). Cache 23499.38 us, SPM 21296.35 us,
1.1034x at default 8 L1D credits, pre-FA-50 software. Not a final comparison: Qwen's
8->16 credit change collapsed its FA lead to ~1.015x; cache L1 credit-full 300.5M vs SPM
154.2M here. Local bounded search only. Review: results/ordered_fa_followup_20261005/
llama/FINAL_REVIEW.md. Suggested (not launched): winner at 16 credits (+flag 15).

**OPS-25 host reboot (2026-10-06):** squire rebooted 10:32 PDT. Last resource-monitor
sample (~10:16) showed host MemAvailable 0.28 GB with our user RSS only 3.4 GB, then the
broker heartbeat and monitor stopped. All simulators and coordinators died. Lost in
progress, never accepted: AI-13 FFN T4096 cache/SPM KC1024 16cr (~5 h each); AI-13b FFN
T2048 cache KC512 / SPM KC256; FA-51b T4096 R256 16cr SPM f15, cache f13, SPM f13 (FA-51b
f13 8cr never started). Accepted evidence untouched. Status files still say running and
broker.json holds 7 dead leases (70 GiB) with a stale heartbeat, so no queue can admit.
The heartbeat/monitor ran inside run_spm_only.py (old supervisor; not to be resumed).
Correction: the 12:04 status report treated stale status files as live runs.
No restart performed pending user decision.
**OPS-25 recovery (2026-10-06, user-authorized):** broker.json backed up
(broker.before_ops25_1791315080.json); removed the 7 dead leases (70 GiB) after confirming
each owner PID absent from /proc. old_reserved 6 GiB left unchanged. New standalone
monitor benchmarks/infrastructure/resource_monitor.py (single-instance flock, same
hold thresholds as the run_spm_only.py heartbeat; writes broker heartbeat/hold_new and
spm_only_resources.json every 5 s; never touches leases). Running: heartbeat fresh,
hold_new false, MemAvailable 695.7 GiB, user RSS 1.9 GiB. Monitor does NOT survive a
reboot; restart it after any host restart. No experiment queue restarted yet.

**FA-51c launch (2026-10-06, user-authorized):** after OPS-25 recovery. Dropped the four
FA-51b flag-13 runs (expected effect ~0.5-1.5%, at the layout-noise floor; shared software,
<=0.5% on the cache/SPM ratio). Runs: (qwen) T4096 SPM Bc256/R256 flag15 16cr, the missing
tuned SPM cell (analytic estimate 8380-8440 us -> ~1.015-1.02x vs cache 8560.84 us; best
measured SPM so far R128 8531.32 us = 1.003x); (llama) FA-41 winner Br16/Bc256/R256 at 16
credits, cache/SPM x flags 0/15, exact bits vs FA-41 (4dacfacb..). benchmarks/qwen/fa/
softmax_tune/fa51c.py reuses FA-51 acceptance; Llama shape plumbing verified against FA-41
argv/work/commands (only guest path, flag word, 8 credit overrides differ). 3 simulators.

### FA-52 — Q-only SPM placement (2026-10-06)
User asked to run Q-only. FA-35 never had a Q-only arm (all Q placements paired with O or
V); prior "Q-only ruled out" was inference only. Same-loop FA-35 evidence: head_v 9667.43 us
vs head_qv 10045.59 us (+3.9%, Br16/Bc256/R512, 8cr). Q-only might still help the cache
path by freeing ~32 KiB of L1 per task. New isolated guest q_only_20261006/guest_v1 =
FA-51 guest v2 + mode 12 (qo_worker, Q resident, O/V coherent); diff 5 lines (qresident,
lock, mode range, geometry check, worker routing). One ELF: mode 2 cache control (0 SPM
ways), mode 12 Q-only, mode 3 V-only; T4096 Br16/Bc256/R512, flag 0, credits 8 and 16
(6 runs). FA-51 acceptance unchanged; exact bits 27f14c53..; per-core tasks/updates from
FA-35 head_v, copies Q-only 14336/core. V-only argv equals FA-35 head_v. No T256 gate
(R512 needs T>=4096 in the head layout); target-size oracle + exact bits instead.

**FA-51c qwen finished (complete_requires_review):** 1 accepted, 0 failed; results/cacheflex_flash_attention/softmax_tune_20261005/fa51c_qwen/status.json.

**FA-52 finished (complete_requires_review):** 6 accepted, 0 failed; results/cacheflex_flash_attention/q_only_20261006/status.json.

**FA-51c llama finished (complete_requires_review):** 4 accepted, 0 failed; results/cacheflex_flash_attention/softmax_tune_20261005/fa51c_llama/status.json.


### OPS-26 — Post-reboot state audit and bounded repository cleanup (2026-10-07)

User requested updating repository state and cleaning erroneous build artifacts.
Host rebooted October 7 at 11:47:17 PDT. Account process inspection found no live
experiment, build or standalone resource monitor. AI-13/13b/FA-51b recorded active
PIDs are absent; their status records predate boot. FA-51b was partly superseded by
FA-51c; dropped flag-13 cases remain dropped. Broker has no leases, 6 GiB historical
old_reserved, hold_new=true and stale heartbeat. No process restarted, no broker or
coordinator-owned state edited. Status CLI now labels pre-boot records, reports PID
visibility without claiming identity verification, and flags stale broker heartbeat.

Registry: 634 accepted records, 70 queues, zero marker/timing discrepancies (not a
new raw-oracle validation). FA-51c Qwen 1, Llama 4 and FA-52 6 accepted; all retain
complete_requires_review. FA-41 final review is already recorded above despite its
old awaiting_final_review status label. Exact timings and links: CURRENT_STATE.md.

RAM audit: 96,192 saved samples, no user RSS above 150 GiB; maximum 115.841457 GiB
at 2026-10-02 05:10:57 PDT, resource_history.jsonl line 82130 under
results/parallel_memory_recovery_20260929. Does not prove a continuous historical
cgroup peak: RSS excludes some cgroup charges, samples have gaps, and boot resets
kernel history. Account kernel CPU/RAM limits remain unlimited; proposed 150 GiB
hard quota was not installed. Operational broker cap remains 150.987 GiB.

Cleanup screened 470 input/manifest files and live open/mapped paths. Archived then
removed 292 source-backed local bytecode files (2,131,309 bytes; net allocated space
reclaimed 1,667,072 bytes). Retained all gem5/incremental/isolated builds, results,
executables, environments, toolchains and unrelated changes; no loose object/library
or simulator files were found in the inspected non-results workload/tool trees.
302 unique FA-41/51c/52 frozen inputs matched manifests before and after cleanup.
Recovery, original documentation, hashes and host snapshot are under
archive/workspace_cleanup_20261007/. No new experiment or performance claim.


### FA-53 — Passive phase breakdown of latest Qwen SPM winner (2026-10-07)

User requested the latest winner's breakdown following the LUT-ingress discussion.
FA-49 covers only the older T1024 diagnostic guest; FA-51c Qwen T4096 winner
8398.4216 us has no phase timing. One exact T4096 Br16/Bc256/R256 flag15 16-credit
SPM replay, 14Q/2KV,d64,8cores. Existing FA-49 observer binary; exact FA-51c guest
unchanged. Disassembly-audited 25-PC map uses existing labels and call/epilogue
boundaries to distinguish P packing, PV dot-product and output accumulation.
No guest instructions added and no rebuilding over retained binaries.
Acceptance reuses FA-51 numerical/config/work/copy/credit/LSQ checks; full independent
oracle plus exact output bits and ALL non-host statistics versus accepted FA-51c.
Per-core phase partition/counts and memory-request conservation/reconciliation
required, plus frozen-input checks. Source benchmarks/qwen/fa/winner_profile;
results/cacheflex_flash_attention/winner_profile_20261007. Registered in current
result log. Decision-changing outcome: where full-ROI critical-core phase time
concentrates on the optimized winner, before committing to an ingress LUT design.
No speedup claim from phase fractions. Host syntax/map-negative/exact-command tests
PASS. Restore standalone OPS-25 monitor after checking no current instance; broker
has zero leases (6 GiB old_reserved retained). One simulator, two shared-affinity
CPUs, 10 GiB lease/8 GiB initial AS plus admitted growth; shared20%CPU/RAM and64GiB
host headroom. No old queue resumed. Accepted control reused instead of duplicate OFF.

**FA-53 complete:** exact FA-51c Qwen SPM winner replay accepted; all non-host stats and output bits match. Full oracle/config/work/copy/resource/frozen-input and phase checks pass. See results/cacheflex_flash_attention/winner_profile_20261007/REPORT.md and FINAL_VERIFICATION.json.

**FA-53 review:** 8398.4216 us, last-arriving core2; softmax38.487%, PV dot product/setup24.717%, QK/max19.863%, Ppacking5.655%, Vstaging3.879%, output conversion/rescale/accumulation2.580%, Qpacking/init2.034%, normalize1.364%, Kpacking0.690%. 2,261 frozen inputs and 3,670,016 outputs checked (zero oracle errors); per-core phase entry counts/partitions and request accounting pass. Existing fused PV function split only at passive retirement boundaries, not independent kernel costs. Final chart visually reviewed; registry635accepted/71queues/0discrepancies. FA-53 lease released; standalone monitor remains available.
