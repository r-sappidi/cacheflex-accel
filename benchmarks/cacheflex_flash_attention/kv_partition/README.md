# Persistent per-core V ownership

This is an isolated software-mapping experiment, not a hardware modification.
Existing FA and SwiGLU queues, guests and simulator binaries are unchanged.

## Hypothesis and controls

Query partitioning repeatedly streams all K/V through each worker. KV partitioning
instead assigns disjoint token ranges of K and V to workers. Each worker stages
its V range once and retains it for all query groups; K remains coherent. This
uses existing L2 ways, not additional backing SRAM, DMA, batch, or multicast.

Both mappings and both memory modes are in the same guest ELF. They share the
previously used 16-row VL16 QK and PV microkernels, vector packing and exp
polynomial. A coherent KV-partitioned control is essential: any improvement
from partitioning alone is not a CacheFlex architectural benefit.

Single-head, noncausal attention, d64, FP16 inputs/GEMM accumulation and FP32
online maximum, denominator, and output accumulator; stored output is FP16.
This does NOT change GEMM accumulation to FP32 or claim exact paper reproduction.
Partitioning changes the rounding schedule; cross-mapping bit equality is not
required, but cache/SPM outputs at the same mapping must match bit for bit.

## Reduction and lifetime

For each query group, worker p computes `(m_p, l_p, o_p)` across its own KV range.
The unnormalized partial output is `o_p`. After a barrier, workers divide output
rows among themselves and compute:

    m = max_p(m_p)
    a_p = exp(m_p - m)
    O = sum_p(a_p * o_p) / sum_p(a_p * l_p)

A second barrier prevents overwriting partials before all reducers finish.
Only one group of partials is allocated, not a T-by-T score matrix. A group has
Br * group query rows. The initial implementation intentionally has a simple,
correct reduction; tree/pipelined reductions are later candidates only if useful.

K packing is parallel and disjoint. All packing, SPM copies, initialization of
online state, QK/softmax/PV, reduction, barriers, normalization, and SPM release
are inside the ROI. Allocation/input generation/pre-touch/export/checking are
outside. This is first invocation after initialization, not guaranteed cold.
Ordinary per-slot SPM RAW tracking orders copied V; no per-panel DSB is added.
End-of-use release is fenced. No observer prints or timestamps inside tile loops;
`kv_phase_*` assembly labels support subsequent PC/trace attribution.

## Geometry and capacity

Main cases: 64 KiB L1I/D, 512 KiB private L2/core, eight 512 KiB LLC banks
(4 MiB total), eight DRAM controllers and fixed eight-router 2x4 mesh. Current
validated Ruby MICRO timing/resource adapter, 1024-entry admission guard with
zero-full-reject requirement, prefetch and LLC retention on. Previous diagnostics'
early-address/forwarding/idealization mechanisms are off. This is not the paper's
exact single-core Classic platform or its two-channel reconstruction.

SPM has a six-way eligibility ceiling, but only touched slots are claimed.
V row: 128 bytes across two adjacent ways at one set. With 1024 sets, rows
0..1023 use ways 0/1, 1024..2047 use 2/3, 2048..3071 use 4/5. Panels must not
cross a way-pair boundary. At T16384/8 cores, V uses exactly 256 KiB/core,
four ways. At T4096/8 it uses 64 KiB/core across half of ways 0/1's slots.

## Queue and acceptance

`run.py` launches at most two additional simulations and stops this suite on
failure. `results/cacheflex_flash_attention/kv_partition_20260925/experiment/`
contains the first launch's evidence. Current queue is `experiment_v3/`, with
frozen hashes, configs, commands, status, all raw results and REPORT.md. Initial
launch exposed a reporting-only issue: single-controller release counters omit
the `::total` suffix. Both scalar and vector-total formats are now checked;
guest, hardware, numerical thresholds and expected counts did not change.
V2 then exposed the config audit's assumption that each CPU serializes a separate
Process object; gem5 instead stores references to CPU0's shared Process. V3
resolves and checks that reference, with two added real-config/negative tests.
Both fixes concern experiment infrastructure, not kernel or simulator behavior.

1. Twelve gates: both mappings and memory modes at 1/4/8 cores. Small-cache
   gates deliberately exercise up to all three way pairs and partial query groups.
2. T4096/8-core screen: query and KV mappings, group rows 64 and 128, Br16/Bc128,
   cache and SPM (eight cases).
3. T4096 query-mapping 1/4-core pairs; combined with the eight-core screen these
   establish strong scaling with identical per-core hardware and fixed fabric.
4. T16384/8-core confirms both mappings, group sizes and backends (eight cases).

Every output is checked against a separate NumPy model with ordered FP16 FMA
rounding and the partition-aware merge. Gates additionally compare to direct
float64 attention (relative L2 < 1%). The rounded-reference per-element tolerance
remains `2e-6 + 1% abs(reference)`. Exact copy/release counts, output hash,
configuration and global-admission guard are required. Reference generation for
large shapes may take time, and status identifies it separately from simulation.

For persistent V, total copies across all cores = 2*T cache lines, independent
of query-group count. For query partitioning it is
`2*T*cores*ceil((T/cores)/(Br*group))`.

These few tile choices are a first screen, NOT a tuned best-to-best claim.
The previous optimized query baseline remains a required comparison on matching
geometry. If that older driver beats this new query control, it must be retained
as the performance baseline; the new common driver cannot replace it silently.
The 1/4/8 scaling here is for this kernel family, not a recovered MICRO figure.

## Build and test

    python3 benchmarks/cacheflex_flash_attention/kv_partition/build.py NEW_BUILD_DIR
    OPENBLAS_NUM_THREADS=1 .venv-fa-optuna/bin/python -B benchmarks/cacheflex_flash_attention/kv_partition/test_reference.py
    OPENBLAS_NUM_THREADS=1 .venv-fa-optuna/bin/python -B benchmarks/cacheflex_flash_attention/kv_partition/test_config.py
    python3 benchmarks/cacheflex_flash_attention/kv_partition/analyze.py results/cacheflex_flash_attention/kv_partition_20260925/experiment_v3

The builder copies and hashes existing kernels and helpers before compilation.
Do not edit frozen build inputs or the running runner. Use a new experiment/build
directory if a technical defect requires repair; retain failed evidence.
