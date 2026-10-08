# Persistent per-core Q, coherent K/V

This experiment replaces the cancelled V-streaming sweep. Query partitioning
is unchanged: each of eight cores owns T/8 Q/O rows. Within a query group,
packed Q is copied once into SPM and remains there through every K/V panel.
Both streamed operands remain coherent. Output/softmax state stays coherent
for this first implementation; an output-resident kernel is NOT implemented.

The QK kernel keeps the existing 16-row pipelined FP16 FMLA schedule, but
replaces each coherent 16-byte replicate-Q load with an aligned 64-byte SPM
load and TBL selection/replication of the relevant 16-byte quarter. Address
alignment and index construction add instructions; these are real costs included
in timings. The PV kernel is unchanged and entirely coherent.

The initial sub-line load attempt hit the frozen Ruby sequencer's DataBlock
response bounds assertion. The aligned-load variant avoids that unsupported
path without modifying the simulator. Failed attempts are retained separately.

### T16K address-bounds repair

V3 passed T512 but failed at tick11628680000 in the T16K run. Frozen replay
`q_crash_trace_20260915` reproduced CPU5 load SN156442 at PC0x402064 with raw
address0xffffffffffffff80 (-128). Ruby canonicalized this to0x6007ff80, way7;
the ISA request is full-VL256B despite the predicate, and spans beyond way8.
This is not insufficient capacity for the valid Q group: valid offsets are
0..8191 in way0. The trace is consistent with a speculative out-of-range
intermediate pointer; the precise initiating branch has not been isolated.

V4 changes the six alignment masks from ~63 to0x1fc0, bounding every computed
address to the 8KiB Q window. Every valid architectural address is unchanged.
Stripped .text comparison shows exactly six changed instructions, identical
section size, and no additional guest instructions. This is a scoped software
address-confinement repair, not a general Ruby speculative-fault handling fix.

`check_q_bounds_fix.py` checks that text invariant and runs the original T16K
case through tick13000000000. That bounded diagnostic is separate from full
numerical acceptance. The V4 runner adds T2048 gates to exercise multiple query
groups, before a full T16K rerun. Use `--spm-only` to avoid duplicating a retained
cache control. Each failed process now writes failed progress immediately;
the coordinator reports failure before waiting for another pilot to finish.

The current 64KiB/8-way L2 has 128 sets and 8KiB per way. Q groups of
16/32/64 rows occupy 2/4/8KiB respectively, in one eligible SPM way. No logical
set folding or multiway remapping is used. Br must be a multiple of 16,
Br*group <=64, and T divisible by 8*Br. Q is overwritten only after completing
all K/V panels for that group; existing address-aware SPM ordering applies.

At T16K, copies total T*d*2/64=32768, independent of group size, versus
8388608 for the earlier V-staging pilot. Reduced copy count does not guarantee
speedup: Q might otherwise hit in L1, TBL costs instructions, and claiming
SPM slots removes coherent capacity.

Entry points: `build_q_resident.py`, `run_q_resident.py`. The new builder
extends the isolated single-head builder; original sources/binaries remain
unchanged. Run from repository root with a fresh output directory:

```sh
.venv-fa-optuna/bin/python -B benchmarks/cacheflex_flash_attention/run_q_resident.py NEW_OUTPUT --sweep
```

First, T512 independent full-reference and observer-off/on exact-counter gates
run for cache Br32/Bc128/G2 and Q-SPM Br16/Bc128/G4. Then the T16K pilot pair
runs concurrently. The optional sweep covers 18 configurations per architecture:
Br16/G1,2,4; Br32/G1,2; Br64/G1, each with Bc128/256/512. This includes matched
cache controls as well as independent ranking; pilots count in the 36 points.
One timing per case, no claim of exhaustive tuning. Additional configurations
run sequentially after the pilot pair validates. Full Bc-specific numerical
checks, output hashes and exact copy counts gate acceptance.

Geometry remains 64KiB L1I/L1D, 64KiB private L2, 4MiB aggregate LLC, VL16.
No hardware changes, DMA, batch or multicast. Progress reporting is the same
simulator activity reporting documented in SINGLE_HEAD.md, not exact percent.
