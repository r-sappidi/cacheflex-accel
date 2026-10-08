# Near-L3 PV extension: analytical feasibility screen

2026-09-29. User asked whether the engines can handle a larger fraction of
arithmetic. This is a design/accounting study, not an implemented or queued
hardware experiment. Related L3-02/03 implement FP32 recurrence and compact FP16
contributions; L3-05 already supplies direct PV epilogues; MOD-09/10 describe
hardware bounds and demonstrate little sensitivity to reduction-engine speed.
FA-06/29 distinguish probability packing from the already implemented PV epilogue.
Ledger/inventory search found no accepted near-L3 PV matrix-product engine.

## Candidate operation

Keep QK, masking and online softmax on the CPU. Move the complete PV tile product
and subsequent output recurrence to the engine:

1. CPU produces probability tile P and per-query scale alpha.
2. Explicitly install P and V in shared L3 SPM, retaining finite copy resources,
   coherence, ownership and source-lifetime rules.
3. A bounded tile descriptor identifies P, V and output slots and their strides.
4. Engine computes the FP16 PV contribution in the existing kernel's numerical
   order, then the separately rounded FP32 alpha*O + contribution update.
5. Existing ordered completion/RAW/drain semantics protect consumers and reuse.

Currently V is private-L2 SPM resident. An L3 engine cannot read it as if it
already occupied its own local bank. Shared V placement/fills, possible duplicate
copies between banks and interference with coherent LLC traffic must be charged.

For Bc256 and head dimension64, one query row performs32768 QK FLOPs and32768
PV FLOPs (FMA counted as two), plus128 recurrence FLOPs for a normal update.
PV plus recurrence covers50.0975% of that arithmetic, versus0.1949% for the
recurrence alone. This excludes softmax and initialization differences; it is
not a predicted latency reduction or fraction of total model arithmetic.

## A throughput barrier for the existing small engine

Accepted T4096 direct baseline:9037.8088us. The busiest core/bank submits64000
row/tile updates; every PV contribution contains256*64=16384 scalar FMAs.
Thus this bank would execute1,048,576,000 scalar PV FMAs. The frozen Ruby clock
is2GHz (500ps). Under an optimistic new design issuing one FMA per lane per
cycle, with perfect reuse, no transfer/stall overhead and full utilization:

| FMA lanes per bank | Compute-only lower bound for busiest bank |
|---:|---:|
|16|32.768ms|
|32|16.384ms|
|64|8.192ms|
|128|4.096ms|
|256|2.048ms|

The current engine has16 FP32 multiply/add lanes, not16 native FP16 FMA lanes.
This calculation already grants it a more suitable arithmetic capability and
ideal issue throughput. A microcode loop over the current engine cannot be
assumed to meet these bounds. Sixteen or32 lanes fail to beat the9.038ms full FA
baseline even with all other costs hidden. Sixty-four lanes leave little room
for overhead.128/256 lanes are screening candidates, not validated efficient or
physically feasible implementations. Eight banks imply eight engine instances;
lane counts do not establish area or power ratios. Changing tiling/mapping would
require recalculating the work bound and independently tuning matched controls.

## Operand movement and local state

For Br16/Bc256/d64:

| Quantity | Bytes |
|---|---:|
|CPU-produced P tile, FP16|8192|
|Current CPU-produced PV contribution tile, FP16|2048|
|V tile, FP16|32768|
|FP16 PV intermediate accumulator tile|2048|
|FP32 output accumulator for these16 rows|4096|
|Existing R256 FP32 output residency per bank|65536|

Sending P rather than the finished PV result increases that raw operand payload
fourfold before headers and alpha. V traffic is additional and placement-dependent.
The existing256-byte capture ABI cannot silently become an8KiB packet. Shared
SPM staging plus a small bounded descriptor is a candidate interface; all staging,
translation, command, drain and buffer costs must be explicit and timed.

Reuse of a loaded V segment across multiple query rows is essential for a larger
engine. Independent per-row engines would repeatedly read the same V data. The
shared bank port remains64B/cycle and serves normal requests too; an engine may
not assume a separate unlimited SRAM port. Internal operand broadcasts and
accumulator registers need a physical cost/timing model. If the intermediate
accumulator instead lives in existing cache SRAM, charge those accesses and
capacity; do not count that storage as free or remove its bandwidth cost.

## Numerical contract

Current CPU PV uses FP16 FMLA and FP16 accumulation across keys, then FP32
nonfused output recurrence. Substituting FP32 GEMM accumulation or separate
half multiply/add changes rounding. A faithful first comparison must implement
the same operation/order, with independently checked full outputs and exact
same-tiling bits. A precision change requires separate matched CPU controls and
an explicit accuracy study; it cannot be hidden inside the hardware speedup.

## Decision before implementation

Build a finite tile schedule/bandwidth model for64/128/256 FP16 FMA lanes and
explicit P/V placement/reuse, with unchanged bank port and coherent capacity.
Reject designs whose compute or compulsory-transfer lower bound already exceeds
the accepted FA runtime. Passing a lower bound is necessary, not sufficient.
Only a candidate with credible margin warrants simulator implementation and
small arithmetic/ordering/resource gates before full FA.

The decisive result is lower complete FA ROI than independently tuned private
SPM and cache, including P/V staging, synchronization, output normalization and
drain. A same-hardware engine-disabled control separates shared-residency effects
from arithmetic offload. Faster GEMM in isolation or a larger offloaded FLOP
fraction does not pass this criterion. No favorable speedup is predicted here.

Evidence: accepted diagnostic stats/config at
results/cacheflex_l3_spm/diagnostics_20260929/experiment_v1/runs/t4096_direct_trace;
guest driver.cpp/pv_direct.hpp in results/cacheflex_l3_spm/qwen_gqa_20260928/guest_v4;
trace interpretation in results/cacheflex_l3_spm/diagnostics_20260929/trace_analysis_v1.
