# Capacity-preserving per-way SPM placement

Related ledger entries: FA-03 found set-local displacement in an older K-staging
kernel; FA-06 already swept whole-region offsets and K/V choices; FA-29 identifies
coherent K/O pressure in the current finite-model V-staging winner. This test
changes **per-way physical set distribution**, not operand choice, guest code,
capacity, copies, service latency or queue sizes. MOD-03 is the current model;
MOD-04 limits claims about SRAM banking.

## Intervention

`RubyCache.spm_set_stride=512` on each private L2 computes
`physical_set = (logical_set + logical_way * 512) modulo 1024`.
At Bc512, the same 1,024 slots (64 KiB) occupy one way in every set instead of
two ways in half the sets. The logical interface and vector lane ordering remain
unchanged. Claim, victim selection, migration, lookup, wide load/store and release
use the same mapping. Each way's rotation is bijective. No extra data SRAM,
separate backing, DMA, multicast, credits or ports are introduced.

The added tag-to-set map is simulator location bookkeeping, not a modeled
hardware table. Hardware would derive the set from slot/way bits. The simulation
keeps the existing latency assumption; it does not prove implementation timing,
area or per-way SRAM row selection feasibility. In particular the current model
does not explicitly arbitrate SRAM banks. Positive results would justify that
hardware feasibility work, not bypass it.

## Validation and decision

`scripts/run_spm_set_spread.py` runs at most two new simulators, preserving old
inputs and the unrelated llama.cpp run. It requires:

1. Default-off all-numeric-nonhost-stat and output-bit equivalence.
2. Three bridge/release regressions with spreading, full-set and small-resource
   stress, independent FA arithmetic, same output hashes and copy/release counts.
3. Tracing-on/off equivalence on a small control.
4. T4096 SPM screen, then T16K cache/control-SPM/spread-SPM on current winners.

Primary decision: full end-to-end ROI change versus original SPM, and speedup
versus independently tuned cache. The T16K controls must reproduce saved results
exactly. Bounded 5us residency windows at 25% of the control ROI diagnose retired
coherent reads, actual removals, refills and reuse by set/PC. They are not whole-ROI
coverage or additive removable runtime. Both null and regressive results are
recorded; reduced evictions without lower runtime is not a performance win.

Old finite simulator and results remain intact. Before editing, the three modified
source files were copied under `results/cacheflex_flash_attention/spm_set_spread_20260927/preserved`.

## SwiGLU follow-up

AI-06–08 motivate transferring the accepted cache/SPM operator pair to the current
finite profile before accumulator blocking. Existing SwiGLU speedups use an older
model. Keep FP16 operands/FP32 accumulation, first versus repeated-call ROIs and
independent full-chain checks unchanged. Do not combine old-model SwiGLU ratios
with this new FA experiment. This transfer is revalidation, not a new mechanism.
