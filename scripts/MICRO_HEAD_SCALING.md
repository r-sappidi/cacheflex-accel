# MICRO-derived multihead reconstruction and scaling

Runner: `run_micro_head_scaling.py`; builder: `build_micro_head_scaling.py`.
Results: `results/cacheflex_flash_attention/micro_head_scaling_20260921`.
The queue uses two simulator processes maximum and leaves async FA/SwiGLU alone.

## Frozen workload

The pristine MICRO commit `82ac0b9de1ed408ce5cd53067267c8a855b4eef3` specifies
LLaMA Hq32/Hkv8/d64 in `experiments/end2end/scripts/sweep_vl16.sh`.
Use VL16, Br64/Bc384 and upstream SPM automatic query grouping. This tile pair
is one repository sweep candidate, **not an identified published winner**.
No causal mask is introduced; the upstream kernel computes dense attention.

FP16 operands and GEMM accumulators; FP32 softmax/running output; FP16 stored
output. Previously validated dense-lane conversion corrections are force-included.
GCC15.2 matches the repository build script's toolchain version. Full original
corrected source, headers, assembly, compiler commands and hashes are retained.
Post-ROI export timing-loads payload and filename, avoiding the diagnosed Ruby
functional-read visibility issue. Guest export hashes are checked on the host.

SPM stages K **and V**, but the upstream VL16 PV routine reads coherent V.
That behavior is deliberately preserved for reconstruction. It is not the
previous custom V-only kernel, and this queue is not an optimization sweep.

## Core mapping and timing

The original corrected single-core driver is a separate measurement. The
parallel harness extracts its per-head body unchanged except for loop bounds.
At 1/4/8 cores, workers own contiguous complete KV groups and associated query
heads: 32/8/4 query heads per worker. Workspaces are private; Q/K/V are shared
read-only arrays and each worker writes disjoint O heads. Each KV group retains
the original per-core transpose/packing reuse. No multicast or DMA is used.

The common-harness ROI includes packing, all head computation, and start/end
barriers; thread creation, workspace allocation and output export are excluded.
Original per-phase clock calls remain. Phase sums are not causal latency costs.
The upstream SE cache flush function is a no-op, so report **first invocation**,
not a guaranteed cold cache. Original cache also performs a post-ROI warm pass;
its warm timing is never substituted for the first ROI.

## Hardware

Use frozen micro_resources/build_v5 and its MICRO-derived miss/writeback pools.
Per core: 64KiB L1I/L1D, 512KiB eight-way L2, four eligible SPM ways for SPM,
none for cache. CacheLoad/Store ports2/1, SPM ports1/1. Core widths5/8,
ROB128/IQ80, regular LQ32/SQ48, upstream FU indices0/2/5/7/9 count2.
VL16; CPU2.5GHz, system1.5GHz. Global sequencer1024 is a bookkeeping guard;
any full-admission rejection fails acceptance.

Shared resources do not grow with cores: eight512KiB LLC banks (4MiB total),
pooled32 LLC miss credits, two DDR4-2400 x64 memory channels, 16GiB memory,
and a fixed 2x4 mesh. Private hierarchy i and LLC bank i attach at router i;
directories attach at0/4. Added cores occupy previously unused private endpoints.
Ruby creates one memory controller per directory, so setting **num_dirs=2**,
not merely mem_channels=2, is necessary. Source MemorySize strings are quoted.

LLC retention off. No extra SPM storage. Ruby prefetch algorithms, controller
pipelines and mesh remain different from the Classic reference; this is not
cycle-identical or calibrated Classic reproduction. The new FU and memory
settings also mean prior geometry-only results are not direct controls.

## Ordered queue and gates

1. T128 Hq32/Hkv8 arithmetic/config checks for original one-core drivers and
   common harness at1/4/8 cores. Compare complete exported outputs to originals.
2. Original corrected one-core T4096 cache/SPM pair.
3. Common-harness one-core T4096 pair, to expose wrapper timing differences.
4. Common-harness four-core then eight-core T4096 pairs.
5. Eight-core T8192 then T16384 pairs, keeping heads and shared hardware fixed.

All outputs are checked against an independent sequential-FP16 GEMM / FP32
online-attention reference with mathematical exp. Small gates additionally
check float64 dense attention. No tolerances are relaxed. Configuration,
worker completion, resource occupancy and source/binary hashes are checked.
Failure stops peers and later stages, preserving logs. Each case has command,
config, raw stats, exports, reference progress and accepted/failed metadata.
`status.json` and `REPORT.md` identify completed cases; a launched run is not
an accepted performance result. Larger-T simulations and numerical references
can take substantially longer than single-head diagnostics.
