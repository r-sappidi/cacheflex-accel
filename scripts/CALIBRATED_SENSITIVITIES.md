# FA sensitivities with calibrated MICRO timing

Runner: `scripts/run_calibrated_sensitivities.py`.
Results: `results/cacheflex_flash_attention/calibrated_sensitivities_20260922`.

The question is whether earlier sensitivity conclusions change when cache/SPM
timing is corrected. Every intervention is run with both the old and calibrated
timing profiles on the **same current Ruby simulator and resource policy**.
This avoids attributing changes from historical model repairs to timing alone.

Fixed:8 cores, VL16, single head, d64,64KiB L1I/D,64KiB privateL2/core,4MiB LLC.
Cache uses Br32/Bc128/G2; SPM uses Br16/Bc128/G4 and two ways. Guest/ROI bytes,
finite miss/writeback pools, retention policy except its named ablation, CPU,
and precision stay fixed. No separate SPM backing, extra cache capacity,
DMA/multicast or tile retuning.

| Intervention | Exact change | Interpretation |
|---|---|---|
| demand_isolation | Existing copy-stalled ingress demand bypass | Arbitration sensitivity; search circuit cost unmodeled |
| ingress4 / ingress16 | Look ahead4/16 FIFO entries | Same transition/queue budgets |
| copy_service | Copy forwarding and SPM bridge response1 cycle;8 SPM store ports | Prior diagnostic service/issue bundle, not a costed design |
| load_issue | SPM-load forwarding1 cycle;8 SPM load ports | Prior issue/forwarding bundle |
| copy_load | Both bundles | Interaction, not sum of independent savings |
| load_service | Pure SPM response1 cycle only | Additional latency-only check, no port expansion |
| no_retention | Disable silent-fetch LLC retention | Policy sensitivity |
| restrained_prefetch |2 streams; train after8 misses | Match previous restrained setting |
| no_prefetch | Prefetch disabled | Traffic/latency tradeoff |
| network256 | Links128→256 bits on both cache/SPM | Explicit hardware bandwidth/cost change |
| guard4096 | Global bookkeeping guard1024→4096 | Must reproduce every finite non-host ROI counter exactly |

The artificial historical16-entry global cap is not reinstated: this experiment
retains the translated hardware-resource pools and tests the nonbinding guard.
Narrowing that guard would reintroduce the model issue already removed.

## Gates and reporting

1. Reuse four fully accepted T4096 controls, with explicit provenance. Their
   hashes are frozen and numerical outputs rechecked. These are not called new runs.
2. Rerun T512 controls for both timing profiles/backends. Require exact ROI
   counter equality, which also checks non-perturbation of copy aggregation.
3. Run all12 interventions at T512 under both profiles/backends (48 gates).
4. After all52 gates pass, run the48 corresponding T4096 cases, two at a time.

Each new case checks every output against the unchanged independent oracle,
full old-output hash, guest/export hash, exact copy counts, all8 present/zero
global admission rejection counters and every configuration field. Only the
declared intervention fields and output path may differ from its profile's
control. SPM runs also check aggregate copy accounting against per-core
outstanding-area counters. No huge per-transaction traces are collected.

The main metric is **end-to-end FA ROI latency**. `REPORT.md` reports same-profile
SPM runtime reduction and matched-platform cache/SPM ratio. Each intervention's
old/new profile comparison shows whether its sensitivity changed. Per-copy
aggregate lifetimes are auxiliary evidence, not additive FA runtime fractions.

`status.json` lists active PIDs and accepted/reused cases. `gates_passed.json`
appears only after all52 small gates. A failure stops peers/downstream runs and
preserves evidence. `inputs.json` freezes dependencies; do not edit them while
the queue is active. Choose a fresh `CACHEFLEX_SENSITIVITY_OUT` for a rerun.

Approximately100 new simulations are queued (52 small and48 full). Based on
the measured~12-minute calibrated T4096 host runtimes, allow several hours,
roughly5 hours at two simulations concurrently; this is not guaranteed.
