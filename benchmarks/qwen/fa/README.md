# Qwen FlashAttention experiments

Causal, 14 query heads / 2 KV heads, dimension 64. Read the ledger for exact
geometry, precision, hardware and acceptance scope. Historical runners are not
safe default launch commands. Check `python3 benchmarks/status.py` first.

| Question | Source directory | Ledger |
|---|---|---|
| Placement and Br/Bc/R tuning | `matrix_runner.py`, `matrix_plan.py` (legacy aliases) | FA-35 |
| Optimistic K/V rotation | `ring/` | FA-34 |
| Matched V-only rotation | `v_ring/` | FA-36 |
| DSB attribution | `dsb_diag/` | FA-37 |
| Ring overlap, neighbor handshakes, head reuse | `ring_async/` | FA-38 |
| SPM-specific bottleneck ablations | `bottleneck/`, `bottleneck_1024/`, `bottleneck_targets/`, `spm_isolation/` | FA-39/40 |
| Packet latency and credit attribution | `packet_diag/`, `packet_parallel/`, `network_relief/` | FA-42/43 |
| Paced/overlapped V staging and matched controls | `paced_staging/`, `software_confirm/` | FA-44/45 |
| Blocking PV feasibility proxy | `ncc_probe_v5/` | FA-46 |
| Asynchronous PV | `pv_async/` | FA-47 |
| PV interface/probe/fence diagnosis; startup-matched controls | `pv_diag/`, `pv_diag_control/` | FA-48 |
| Passive per-core phase and memory-overlap counters | `phase_profile/` | FA-49 |
| Exact-bit softmax/PV software options | `softmax_opt/` | FA-50 |
| Bc/R tuning and 16-credit Qwen/Llama follow-up | `softmax_tune/` | FA-51/51c |
| Q-only versus cache and V-only placement | `q_only/` | FA-52 |
| Exact latest Qwen SPM winner phase replay | `winner_profile/` | FA-53 |

`ncc_probe/` through `ncc_probe_v4/` preserve superseded implementations and failed
attempts. Do not use their timings as accepted FA-46 results. Frozen versions stay
in place because successor builds and evidence can reference them.

## Evidence

- Current 4K matrix: `results/retained_protocol_restart_20261002/fa_matrix_spm_only/`.
- Latest phase report, CSV, chart and final audit:
  `results/cacheflex_flash_attention/phase_profile_20261005/`.
- PV diagnosis: `results/cacheflex_flash_attention/pv_diag_20261005/`.
- All queue locations: [current result log](../../../docs/experiments/CURRENT_RESULTS.txt).
- Accepted interpretations: [ledger](../../../docs/experiments/LEDGER.md).

`reference.py` and `kv_reference.py` are numerical oracle aliases. The matrix and
long-context result aliases refer to historical roots; consult the current log
for authoritative retained-protocol queues. Long-context work was cancelled by
the user. Shared kernel sources remain in `benchmarks/cacheflex_flash_attention/`.

Phase timing is observational: copy timing starts at Sequencer issue, events use
the current retirement phase, and pending loads do not imply whole-core stalls.
PV plus accumulation stays fused. Optimistic ring/PV results are not finite-link,
finite-bank hardware performance predictions.
