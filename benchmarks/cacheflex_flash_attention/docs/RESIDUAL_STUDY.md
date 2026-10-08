# Remaining FA gap experiments

Active run: `results/cacheflex_flash_attention/residual_20260911`.

The preceding factorial accelerated selected services but left most of the
SPM-minus-cache gap. This follow-up separates software organization, per-core
phase/imbalance, resident QK performance and hardware sensitivity.

## Comparisons

1. Original independently structured cache/SPM guests, freshly run on the
   frozen boundary-observer simulator. Historical guests remain unchanged.
2. Matched cache/SPM guests: same grouped SPM-driver traversal, Br64/Bc128,
   eight heads/cores, d64, native VL16, all-query grouping, two alternating
   banks, allocations, shared K packing, Q/P packing, softmax, PV and barriers.
   Cache reads shared packed K; SPM copies from that same buffer. This changes
   SPM's source layout and adds common K preparation, so it is a controlled
   new pair, not a claim that the original driver was byte-identical.
3. Observer-off/on, **same guest ELF**, at T128 and T1024. Retiring labeled
   instructions produce phase-boundary ticks. The observer must leave all
   extracted timing/counters/output unchanged before results are accepted.
4. Resident QK: exact generated 8x128 kernels, eight cores, 64x128x64 per core,
   private A/B/C buffers, warmup, then 128 repetitions. All SPM copies occur
   before ROI; all output elements have an exact nonzero analytic reference.
   This is a synthetic kernel diagnostic, not end-to-end FA performance.
5. Paired matched-driver sensitivities at T1024: regular load ports, ROB/IQ/
   LSQ/register capacity, Ruby controller transition throughput, sequencer
   outstanding capacity, and SIMD operation latency. Exact settings are saved
   in `sensitivity_parameters.json` and each run's `command.json`/`config.ini`.

Original and matched outputs must have identical finite hashes. This is not
independent numerical validation of the full FA math. No multicast, batch or
DMA is added. Hardware experiments are diagnostics, not area-matched proposals.

## Phase measurements

`cacheFlexBoundaryPcs` defaults to an empty list. When enabled, the commit stage
observes successful retirement of the selected static PCs, only once per
macro-instruction. The `CacheFlexBoundary` debug stream records ticks, PC and
dynamic sequence number without changing architectural or timing state.

The matched source's asm labels encode no guest instructions. However, GCC
optimized empty-asm and labeled-asm builds differently. Therefore they are
**not** used as off/on controls: both simulations execute the same labeled
guest, stripped from its debug ELF. The older empty-asm build is retained only
as a compiler diagnostic. Source-level memory barriers belong to both new
matched variants; this is not retroactive instrumentation of the original ELF.

`boundary_analysis.json` partitions each core's ROI time by retiring-boundary
intervals, including pre-boundary/unlabeled time, loop control and synchronization.
It records the spread in arrival at the terminal barrier and the latest-arriving
core. This exposes imbalance; it does not equate summed core time to wall time
or establish a unique algorithmic critical path through all shared dependencies.
Out-of-order execution crosses retirement boundaries, so these are not exact
execution-unit occupancy intervals or causal stall-cycle bins.

The initial historical PC attribution is saved in `residual_phases_20260911`.
Every debug ELF used there was checked for .text identity against the actual
guest. Those aggregate histograms do not themselves establish causality.

Results are appended to `summary.json`; failures stop downstream work.
`REPORT.md` is generated only after every queued stage passes.
