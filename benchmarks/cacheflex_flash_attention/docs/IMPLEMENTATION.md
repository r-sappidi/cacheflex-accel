# Current FA implementation: reading guide

## Start with the actual build, not the filename

The current corrected search builds **both cache and SPM variants** from
`spm_multicore.cpp`. The name reflects its history, not a restriction to SPM.
`cache_multicore.cpp` is the older separate cache-driver path; it is not the
source of every modern coherent-cache result.

The current build sequence is:

```text
build_pv_improvements.py
  → build_correct_search.py
      → make_search_kernels.py: generate local QK/PV kernel definitions
      → copy spm_multicore.cpp into the result build directory
      → compile with correct_fa_conversions.hpp and cache_qk_wrapper.cpp
  → optionally generate full-width PV and modify the isolated driver copy
  → rebuild the isolated guest
```

Builders perform checked textual transformations. Read the generated
`driver.cpp`, `kernels.hpp`, optional `pv_fullwidth.hpp`, `experiment.json`,
and `build_command.sh` under a result's build directory to identify the exact
implementation measured. Do not infer that solely from the base source.
Avoid reformatting the shared source without updating and testing those
transformations. Existing native assembly is also imported from the sibling
`cacheflex_micro` tree; this benchmark directory is not self-contained.

## Shapes, ownership and dataflow

`T` is sequence length, `d` head dimension, `Br` query rows per compute block,
and `Bc` keys/values per panel. `group` counts query blocks, so a group's row
count is approximately `group × Br`, clipped at the sequence boundary.

In the shared driver, each worker owns a distinct `T × d` slice of Q and O;
workers share the same `T × d` K and V arrays. This is not eight completely
independent heads with distinct K/V, nor the proposed KV-sharded algorithm.
Keep that distinction explicit when discussing model representativeness.

Within each worker, the main traversal is:

```text
query group
  pack Q; initialize FP32 running maximum, denominator and output accumulator
  for each K/V panel:
    stage selected operands (SPM variants; optional overlap)
    for each query block in the group:
      QKᵀ → update online softmax/rescale → pack P → PV → accumulate output
  normalize and store the group's output
```

The interleaved variant instead processes all QK blocks, then all softmax
blocks, then all PV blocks for a panel. Group size determines panel reuse,
scratch footprint and the number of repeated K/V passes; it is not a GEMM
microkernel register-tile dimension.

## Source map: `spm_multicore.cpp`

| Symbol/section | Responsibility |
| --- | --- |
| Configuration macros and kernel includes | Select legacy/native VL16, placement, buffering and diagnostic variants |
| `copy_line`, `dma_2d`, claim/drain helpers | Low-level staging mechanisms; not all are enabled in current VL16 builds |
| `stage_kv` | Copy selected K/V panels into encoded private SPM slots |
| `Shared`, `Ctx` | Shared matrices, per-worker identity and synchronization |
| `worker`: scratch allocation and preparation | Pack Q, prepare K, allocate S/P and FP32 online-softmax/output state |
| `worker`: `qk_phase` | Cache or SPM GEMM computes score tiles |
| `worker`: `softmax_phase` | Update running softmax state and rescale prior output |
| `worker`: `pv_phase` | Pack probabilities, multiply by V and accumulate output |
| `worker`: final normalization/synchronization | Write O and close the measured ROI |
| `main` | Argument/variant checks, inputs, workers and output diagnostics |

Q, output accumulation and softmax scratch remain coherent. Placement selects
cache-only, K-only, V-only or K+V staging; do not describe all SPM runs as
K-staging. Full-width PV is added to the generated copy, not directly to this
base driver. FP16 GEMM operands/results and FP32 softmax/output accumulation
have a specific rounding schedule checked by the independent reference.

## Configuration knobs

| Knob | Meaning |
| --- | --- |
| `placement` | `cache`, `k`, `v`, or `kv` |
| `group`, `Br`, `Bc` | Reuse group and outer compute panels |
| `overlap`, `offset`, `pinned` | SPM buffering/slot layout and special K residency variant |
| `rows`, `pipeline`, `unroll`, `order` | QK kernel shape and instruction scheduling |
| `fusion`, `interleave` | Output-rescale fusion and phase ordering |
| `pv_fullwidth`, `stride_fix` | Additional isolated transformations in the PV builder |

Capacity eligibility must be checked against the actual cache/SPM geometry.
Not every knob combination is supported. Software tuning must keep hardware
fixed; `run_llc_retain.py` is explicitly a different hardware experiment.

## Reading a result responsibly

1. Identify frozen simulator/guest hashes and configuration, including total
   LLC capacity across banks, VL, replacement policy and SPM eligibility.
2. Require the numerical gate and complete output export; matching hashes alone
   are not independent correctness validation.
3. Compare the same workload and ROI with independently selected cache/SPM
   configurations. Include preparation/synchronization according to that ROI.
4. Use traffic counts to explain data movement, and phase/PC attribution to
   locate time. Attribution percentages alone do not prove causal saved cycles.
5. Separate architectural controls, diagnostic ablations and accepted
   best-to-best timings. Consult each result's report for acceptance status.
