# Analytical selection for Qwen-shaped causal GQA

User requested the analytical approach rather than broad blind sweeps.
Related FA-09 independently validated traffic/storage counts but failed to
predict runtime from traffic alone. FA-29 used sparse nonnegative fitting and
bounded candidate validation, with17.08% historical held-out error. L3-05 tuned
single-head direct epilogues; L3-06 transfers those to Hq14/Hkv2 causal attention.
New here is exact enumeration of causal padded work and rotated task ownership,
finite L3 accumulator occupancy, and a small controlled parameter follow-up.

Script scripts/qwen_analytical_model.py computes per-core QK/PV pair counts,
SPM copies, accumulator updates/144B transport, tasks, live operand storage,
and private/LLC capacity use. Natural head/query coordinates are independently
mapped back to core ownership. Counts match every accepted source case.
Useful causal work and padded executed work are separate. Full-head arrays,
active per-core state and physical SPM claims are separate. Cache-set conflicts,
actual DRAM misses, instruction overlap and engine stalls are not predicted.

Do not import old noncausal/model timing coefficients. Current Qwen timings
all use R256; an R-dependent regression would be underidentified. Instead,
use explicit nonnegative sensitivity scenarios to rank legal near-neighbor
geometries. The scenario weights are heuristic ranking preferences, NOT fitted
latency fractions or calibrated hardware costs. No predicted-microsecond or
optimality claim. Retain the baseline and explore one orthogonal tradeoff.

Retain Br16. Enumerate R64/128/256/512 and Bc128/256/512/1024 analytically only.
Initial simulation budget defers R64's fourfold staging, Bc1024's prior severe
regression, and scheduler cases with >15% max/mean causal arithmetic imbalance.
These are budget screens, not correctness restrictions or universal exclusions.
R512/Bc256 is deferred at T2048 but remains feasible at T4096; revisit only if
later evidence warrants a size-specific follow-up.

Current selected geometries:
- Br16/Bc256/R128: grouping/ownership and footprint versus doubled staging.
- Br16/Bc512/R256: fewer commands versus larger causal padding/operand footprint.
Each runs on BOTH private-SPM and L3, preserving matched merge comparisons.
The strong inherited cache and additional accepted cache control remain frozen.

scripts/launch_qwen_analytical.py waits for all original L3-06 targets to finish
and their coordinator to exit successfully. It revalidates the counts, saves
the final plan, and derives a separate runner from the frozen L3-06 runner.
The run() acceptance function has an identical AST: numerical, exact-bit,
configuration, command/copy, export, finite-LSQ/queue checks are unchanged.
Only output paths, reference reuse, phase selection, labels and reporting differ.
Same guest_v4 ELF and lifecycle build_v5, same scheduler/resources. No new build.

Budget: four T1024 candidate gates, four T2048 measurements, at most two T4096
transfers (only arms with an actual T2048 ROI improvement). Original accepted
results stay eligible winners. Baseline/reference files are reused with hashes.
Failure stops peers/downstream. All complete outputs independently checked;
same-Bc results must remain bit-identical across R changes. No tolerance changes.
Final report includes negative/null results and measured-versus-ranked feedback.

Decision: reduce complete attention ROI versus each arm's inherited control,
then compare strongest measured private-SPM versus L3 and frozen cache. Include
packing, staging, causal masking, normalization, barriers and drain/release.
No speedup from lower traffic alone. No full-model/BF16/area/energy inference.

Max2 new simulations, launched only after current Qwen simulations complete;
existing SwiGLU/llama untouched. Same nice10, CPU>=50%idle, memory>=128GiB,
disk>=100GiB and total live-simulator launch guards as L3-06. During runs,
memory>=64GiB and disk>=75GiB guards remain. Single-thread reference generation.
Results: results/cacheflex_l3_spm/qwen_analytical_20260928.
