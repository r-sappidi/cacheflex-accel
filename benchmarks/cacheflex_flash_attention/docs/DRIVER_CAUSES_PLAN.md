# Remaining FA slowdown: causal driver study

Current workload: T1024, d64, Br64/Bc128, VL16, eight independent query
heads (one per CPU) sharing one K/V head, Hq8/Hkv1. This is not one head
partitioned across eight CPUs, a full attention layer, or a full model.
Existing finite periodic-output checks are regressions, not independent
full-attention numerical accuracy validation.

Reference implementations and repaired simulator are preserved. Default
hardware, scalar SPMCP, K-only staging, no multicast/batch/DMA. The starting
software points are original cache, matched cache group1, matched SPM group4.
The previous all-query hardware sensitivities are not attributed to these winners.

Stage 1: independently switch each concrete driver difference, with shared
baseline/SPM implementations and configuration/hash/copy-count checks:

- P packing: legacy source loop versus original cache's explicit SVE transpose.
  GCC does auto-vectorize the legacy loop, but adds alias/version checks;
  inspect emitted instructions and measured function counts, not source alone.
- Q packing: separate switch for original packer (d64 is below VL16's 128
  FP16 elements, so it takes the explicit packer's scalar tail).
- Shared K packing: vector row copies/padding versus current source loop.
- Startup synchronization: remove the second consecutive barrier (no multicast).
- Restore original cache's Q-tile copy before Q packing.
- Cache QK: eliminate the out-of-line wrapper while retaining assembly body.

Run the complete 3-bit packing factorial to account for interactions. Obtain
identical-guest boundary observer controls plus function/retired-PC attribution
for control, P-only, all-packing. Test vector packing independently bit-for-bit
on nonperiodic source bits including tails/padded rows. Code-generation and
memory-layout effects are possible; no phase delta is automatically a causal
execution-unit stall penalty.

Stage 2 (chosen from stage-1 evidence): repeat group sensitivity with repaired
driver functions; apply interventions to both architectures and retain original
cache in winner selection. Examine last-core phase changes, retired instruction
counts, copy counts, cache/NoC traffic and working-set sizes together.

Stage 3: on the resulting software finalists, test targeted capacity/ordering/
overlap counterfactuals where an unexplained gap remains. Reuse only implemented
and audited simulator knobs, label hypothetical hardware interventions, and do
not count them as default-hardware software speedups. Use residency traces for
buffer-specific corroboration when locality is implicated. Quantify interactions
and unexplained residuals rather than force an additive attribution.

Completion requires a measured mechanism-specific explanation of the gap,
validated changes in the implicated work/traffic, plus an explicit residual
bound. A completed queue alone does not satisfy this objective.
