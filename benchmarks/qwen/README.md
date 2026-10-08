# Qwen-shaped component benchmarks

These are component benchmarks, not an integrated end-to-end model execution. The FA workload uses 14 query heads, 2 KV heads, head dimension 64 and causal attention on eight simulated cores. FFN and projection timings must be combined only under the evaluation methodology recorded in the experiment ledger.

| Component | Entry point | Evidence |
|---|---|---|
| FA tuning and diagnostics | [FA index](fa/README.md) | FA-34–52; current result log identifies retained queues |
| FA long context (cancelled) | Historical `fa/long_context_runner.py` | Do not restart without user authorization |
| SwiGLU FFN | [ffn/runner.py](ffn/runner.py) | `results/ai_workloads/`; consult accepted reports and ledger |
| QKV/output projections | [projections/runner.py](projections/runner.py) | `results/ai_workloads/`; consult accepted reports and ledger |
| Remaining pipeline operators | [pipeline/remainder_runner.py](pipeline/remainder_runner.py) | Component-specific accepted reports |
| Component aggregation | [pipeline/collect_components.py](pipeline/collect_components.py) | Projection methodology and validated component records |

FA numerical oracles: [query-owned reference](fa/reference.py), [KV-partitioned reference](fa/kv_reference.py). The [frozen matrix guest](fa/frozen_matrix_guest/) is a historical source snapshot; it is immutable evidence, not a development directory. Reusable source kernels remain in `benchmarks/cacheflex_flash_attention/`.

The matrix is a bounded candidate/neighbor search, not proof of a global optimum. A completed simulation is usable only after arithmetic, output, configuration, and resource checks pass. Historical failed or cancelled subsets must not be mixed with accepted results.

Launchers are linked at their original paths for compatibility. Do not run another instance while a coordinator is active. Inspect `python3 benchmarks/status.py` and the linked status files first.

## Optimistic neighbor-SPM transfer

FA-34 matched ordinary/optimistic K/V delivery with finite buffers and timed
CPU/barriers. This is a headroom diagnostic, not realistic ring hardware timing.
FA-34 and FA-38 bulk files are now in the
[recoverable ring archive](../../archive/ring_mechanism_20261005/summary.json)
(OPS-24). FA-36/37 dependencies remain in place. Archived runners are not launch
instructions.
