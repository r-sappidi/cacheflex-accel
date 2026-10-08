# Bounded near-SPM FA accumulation experiment

## Decision and prior evidence

User-authorized FA-first experiment, September 27. FA-25–27 already tested
output residency, forwarding and early store address resolution without a
competitive best-to-best benefit. FA-29 supplies validated, analytically tuned
query-partition baselines: T16384, eight cores, VL16, d64, Br16/Bc512/R768.
This experiment adds **arithmetic offload plus posted completion**, not just
residency. No extra accumulator array is added.

Primary outcome: end-to-end FA ROI versus the fastest validated coherent and
standard-SPM controls on the same hardware. A substantial runtime reduction,
not fewer retries or shorter transaction lifetime, determines whether to pursue
the mechanism. Null results constrain this implementation, not all near-cache
reduction designs. Do not add per-core stall counts into runtime percentages.

## First implementation and explicit limitations

Each core has four source records (256B contribution +16B header each), a
16B/cycle dedicated local command link with two-cycle fixed latency, four sink
records including the active operation, and one active 256B intermediate.
At most 2432B payload/header buffering per core; not a full accumulator SRAM.
Source records remain occupied during transfer; a sink entry is reserved before
transfer begins. Full queues stall. One update executes at a time per core.

An update computes 64 FP32 elements `O = round(round(alpha*O) + contribution)`.
Sixteen lanes/cycle, three-cycle multiply and add latencies give nine arithmetic
cycles per row. SoftFloat supplies round-to-nearest non-fused arithmetic.
This experimental ABI does not expose arithmetic exception flags to software.
Only unmasked 2048-bit FP32 rows are supported; unsupported commands fail loudly.

For this conservative first backend, engine reads and writes traverse the
existing timed pure-SPM controller queues and share their admission/arbitration
with ordinary SPM accesses. No functional read/write bypass is used. The backend
retains local-hop/response overhead a fully integrated SRAM-side engine might
avoid. Existing gem5 does not model detailed per-bank SRAM arbitration; this
experiment **does not establish a calibrated physical bank/port implementation**.
The new finite ALU and link are additional hardware, explicitly priced in timing
and storage, not purportedly free CacheFlex resources.

SPM ways0/1 hold the original V tile; ways2–5 hold up to1024 FP32 output rows.
Claim/copy, normalization, drain and release costs stay inside ROI. All claims
use normal SPMCP and existing coherent eviction handling. Coherent capacity is
therefore actually displaced; extra backing is prohibited.

Commands use an opt-in experimental SPM-store address encoding: low19 bits slot,
bits51:20 FP32 scale, bit60 command. The scalar header and contribution pass
through real CPU register/store paths; no host-side bulk computation or DMA.
This is a prototype ABI, not a finalized architectural ISA encoding.

Pending destinations are tracked from capture until installation. Ordinary
SPM reads/writes, bridge overwrite/writeback and release cannot pass overlapping
pending updates; commands wait for older accepted accesses. Commands execute
FIFO in acceptance order. Early ACK releases the CPU packet only after copying
its payload. Architectural barriers and CPU/Ruby drain still see posted updates.
Full-system context switching, checkpoint restore of active commands, alternate
rounding modes, predicates and simultaneous DMA/multicast are outside this gate;
do not claim general ISA/OS support.

## Controls and acceptance

1. Same old binaries, new simulator with engine disabled: output hashes and all
   prior finite numeric nonhost stats must match the frozen repaired model.
2. Directed tests: saturation, capture lifetime, per-destination recurrence,
   immediate RAW, ordinary overwrite, SPMWB, DSB, release and slot reuse.
   Test both completion policies and one-entry queues/slow service.
3. Same new FA ELF: original driver mode0; software-fused coherent accumulator
   mode1; software-fused SPM accumulator mode2; reduction mode3 with synchronous
   and asynchronous completion. Cache and normal V-SPM get mode1 controls too.
4. T1024 full independent arithmetic and exact output-hash checks, then T4096
   screening, then T16384 confirmation. Keep the better old/new software control.
   All modes preserve QK/PV microkernels and non-fused FP32 arithmetic.
5. Accepted/completed commands and expected operation/copy counts must agree,
   bounded occupancies must respect configuration, and global sequencer guard
   must remain nonbinding. No downstream performance acceptance after failures.

Snapshots and outputs: `results/cacheflex_flash_attention/near_spm_reduction_20260927`.
Existing active SwiGLU and llama.cpp runs and all frozen inputs remain untouched.
