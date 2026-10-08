# Cache FA optimization on the persistent-V hardware

No SPM or simulator changes. Existing KV experiment/queues remain frozen.
Uses the original VL16 16-row FP16 QK/PV microkernels and FP32 online state.

One software change: the old generic Q packer has a scalar tail when reduction
length64 < VL16's128 FP16 lanes. `build.py` specializes the SAME permutation
network with predicated64-element loads and only four valid output stores.
Both original and optimized guests are retained. Full output bits must agree
in three original/optimized T1024 gates at Bc128/256/512; both also pass the
independent rounded reference and dense float64 accuracy check.

After six gates, 14 T4096 cache configurations screen:
Br16/32 x Bc128/256/512 x query-group rows128/256 (12 cases), plus Br16,
512 query-group rows with Bc128/256 (2 cases). Query groups = group*Br.
The best four go to T16384, alongside fixed larger-reuse corners and a
packing-only ablation at the winning tile. Duplicate cases are removed.
The T4096 ranking is only a screen, not assumed valid at T16384. Original
accepted G4/G8 custom cache measurements remain eligible baseline results.
This is a bounded search, not exhaustive global optimization. The historical
different-driver cache winner is still an additional comparison to consider.

All output values checked independently with a per-Bc rounded reference.
Same exact hardware/config audit as llama.cpp and existing KV experiment;
all packing, computation and synchronization inside first-call ROI, no claim
of guaranteed cold caches. At most two new simulations. Numerical/config
failure halts this suite without stopping other active work.

Artifacts/status: `results/cacheflex_flash_attention/cache_tuning_20260925/experiment/`.
