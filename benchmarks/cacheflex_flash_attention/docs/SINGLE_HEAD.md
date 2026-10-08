# Single-head query-parallel experiment

`build_single_head.py` transforms an isolated corrected/full-width-PV driver,
preserving the existing multi-Q-slice baseline. There is one T×64 Q, K, V and
output matrix; eight workers each own contiguous T/8 query/output rows and
read the complete shared K/V matrices. No output reduction is required.
Currently T must be divisible by 8×Br. Attention is noncausal, FP16 GEMMs
with FP32 online-softmax/output state, native VL16.

From the repository root:

```sh
.venv-fa-optuna/bin/python -B benchmarks/cacheflex_flash_attention/run_single_head.py NEW_OUTPUT --tokens 16384 --sweep
```

Without `--sweep`, only smoke controls and the two pilots run. Initial settings:

| Architecture | Br | Bc | Query blocks/group | Placement |
| --- | --- | --- | --- | --- |
| Cache | 32 | 128 | 2 | coherent |
| SPM | 16 | 128 | 4 | V-only, single buffer, zero set offset |

Both pilots keep 64 query rows of running state per worker. Hardware has
64KiB L1I/L1D, 64KiB/8-way private L2, and eight 512KiB LLC banks (4MiB total).
SPM enables two ways and stages 16KiB of V at Bc128. There are only 128 physical
sets: larger V panels would alias under this existing layout. No DMA,
multicast, batch copies or LLC-retention modification is enabled.

The optional sweep contains 36 cache points (Br16/32/64 × Bc128/256/512 ×
G1/2/4/8) and 12 SPM points (same Br/G, Bc128). Pilots count toward those 48.
The pilot pair runs concurrently, followed by sequential sweep cases.
This is an initial bounded search, not a claim of optimal SPM placement.

## Correctness and progress

Before pilots, T512 observer-off/on runs require identical output and extracted
simulated counters for each architecture. Outputs are exported outside ROI,
checked against the guest hash, then checked in full against a Bc-specific
independent single-head numerical reference (2e-6 + 1% relative tolerance).
SPM issued/completed copy counts and actual instantiated cache sizes are checked.

`status.json` records queue stages and accepted results. Each run writes
`progress.json` with host elapsed time, startup/ROI status and recent per-core
committed-instruction events. These use existing gem5 CPUProgressEvent at
10µs simulated intervals, adding no guest instructions. They are activity
indicators, **not exact panel completion percentages or reliable ETAs**.
Reference computation can continue after simulation finishes; acceptance is
reported separately. `measured.json` is not yet an accepted numerical result.

The inherited guest result string may still say `spm_private` for a coherent
build; use the recorded case/config/placement, not that legacy label. Frozen
command lines and simulator identify the actual architecture.
