# FA causal timing study

This study addresses the native-VL16, eight-core, T1024, Br64/Bc128 FA
configuration previously measured by the residency/capacity probe. It does
not retune the kernels or claim an exhaustive evaluation of FlashAttention.

## Interventions

Each intervention is independently switchable. All 16 combinations are run
for both coherent-cache and scalar-SPM guests. No multicast, batch or DMA
instructions are introduced; original guest bytes and arithmetic are retained.

| Factor | Intervention | Costs it does **not** remove |
|---|---|---|
| Copy service | SPMCP request enqueue 2→1 cycles; bridge acknowledgement 6→1; SPM store ports 1→8 | Copy instructions, source fetches, ownership transfer, frontend bandwidth, dependencies |
| SPM load service | SPMLD request enqueue 2→1 cycles; SPM load ports 2→8 | One-cycle response, controller/FIFO arbitration, RAW/WAR requirements |
| Coherence transport | NoC width 128→1024 bits; private-L2/LLC request and response enqueue delays 2→1 cycles; directory delay 6→1 | Number of protocol transactions, router/link hop latency, arbitration and controller throughput limits |
| Capacity | Four additional SPM-only ways, retaining all eight coherent ways | Other management costs; extra SRAM is not an area-matched design |

Cycle delays are in the corresponding Ruby controller clock domain. Width is
changed consistently using the existing Garnet configuration option, not by
changing packet contents or dropping traffic. The copy-response delay is the
existing bridge-operation parameter; this scalar K-only guest uses SPMCP
rather than SPMWB/multicast. The baseline receives identical parameter changes
(SPM-only parameters are inactive when it issues no SPM instructions).

Only two new hardware parameters are necessary:
`L0Cache.spm_load_request_latency` and `spm_copy_request_latency`.
Both default to zero, meaning **inherit** existing `spm_request_latency`,
not zero-cycle service. All other interventions use existing parameters.
No coherence transition or queue routing is changed. Never infer that this
study measures the benefit of removing all staging/coherence overhead.

Requests share an ordered L0→L1 buffer, so independent shorter service delays
can otherwise produce illegal out-of-order arrival timestamps. Opt-in
`MessageBuffer.clamp_fifo_arrival` delays such arrivals to the prior arrival
timestamp; existing sequence-number tie breaking retains FIFO order. It is
enabled only on that buffer when copy/load request timing is shortened.
It defaults off and retains the original assertion elsewhere. This does not
bypass ordering or provide a new dedicated request channel.

## Validation and execution

`run_causal.py NEW_OUT NEW_SIM OLD_SIM` freezes simulator/guest bytes and
analysis sources. OLD_SIM must include the MLP LSQ duplicate-wakeup repair,
so that the initial equivalence check isolates only these new knobs.

The queue performs:

1. T128 old/new default equivalence, both cache and SPM. Require equal output
   and every extracted timing/traffic/core counter, with only the two new
   disabled parameters allowed in the hardware config.
2. T128 single-factor and all-on correctness/configuration tests.
3. T1024 full factorial: 16 masks × two architectures.
4. Two extra repetitions per architecture of the default and all-on endpoints.
5. Automatic `attribution.json` and `REPORT.md` generation.

Every run checks normal exit, finite matching output hashes, all issued copies
completed, and a field-by-field configuration whitelist. Hash agreement is
functional regression testing, **not** independent numerical validation.
Failures stop the queue. No source changes are permitted while it runs.
The runner reuses archived command lines and does not edit the shared FA/MLP
runner, so existing frozen MLP jobs can continue. FA concurrency is two.

## Meaning of the additive report

For each factor mask S, define the measured wall-time gap:

`gap(S) = SPM_time(S) − cache_time(S)`.

The report provides standalone changes, pairwise interactions at default
settings, and Shapley allocations over all factor orderings. Contributions
are the average marginal gap reductions across these orderings. They satisfy:

`original_gap = sum(allocated_reductions) + all_on_residual_gap`.

Negative contributions remain negative. They mean the intervention hurt SPM
relative to cache; they are not clipped or forced into positive percentages.
The residual is **unexplained by these interventions**, not automatically
"compute time". It includes software/layout effects and mechanisms only
partially accelerated by the interventions. Allocations depend on the chosen
factor boundaries and endpoints; they are not intrinsic per-instruction costs.

Endpoint repeats expose observed variability but are not confidence intervals
for the entire factorial. Effects comparable to variability need additional
repetitions of the relevant pairs. Existing service-time, traffic and ROB-head
PC counters remain supporting observations, not additive causal stall counts.

Tests: `python3 -m unittest discover -s benchmarks/cacheflex_flash_attention -p test_causal.py`.
