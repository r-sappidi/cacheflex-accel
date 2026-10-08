FA-39: identify SPM-specific limits, not merely general FA headroom

Sources are isolated; existing guests, simulator binaries and queues are immutable.
Guest derives from accepted matrix guest_v3; retained-protocol simulator reused.
No hardware timing/queue implementation or functional-model changes.

Twenty-five serial cases: exact old private T256 control (all finite nonhost stats
and output bits), then 12 T256/Hq14 gates, then 12 T4096/Hq14 target comparisons.
Each size: cache and private-SPM for control, staging specialization, L1D credits16,
privateL2 credits40, both, and direct packed-P producer. Original credits8/20.
All cases Br16/Bc256/R256,Hkv2,d64,causal,eight cores,FP16 arithmetic/FP32 state.
One new ELF carries runtime flags for both software variants, so every matched
pair uses identical code placement. New controls are required: their timings are
not substituted for the historical matrix winner. Cache stage flag is a no-op
control. Packed-P uses rounded FP16 bits in strided SVE halfword stores; this exact
implementation may lose due to scatter cost. A loss does not prove all fusion bad.

SPM attribution: compare fractional full-ROI reduction within each backend, then
SPM reduction minus cache reduction. Shared hardware changes must be identical
for both backends. Counter checks establish whether pressure was actually relieved;
otherwise a null result is inconclusive. No assumption of a unique bottleneck.
If both backends improve equally, classify as shared FA headroom, not SPM-specific.
Credit2x2 interaction: compare combined versus individual effects; do not sum
overlapping stalls or equate queue-full attempts with elapsed cycles.

Stage specialization keeps exact destination/copy sequence for rows<=sets, with
generic fallback. Fusion keeps per-row softmax/reduction order and FP16 rounding,
directly writes eight-row column-major P layout and bypasses generic P packing.
No Q/K/V ownership, causal tile visits, SPM slot allocation or release changes.
Full independent reference, dense gate reference, exact output bits across all
same-size variants, per-core work, copy issue/completion, zero reductions, finite
LSQ/globalguard and exact permitted config differences required. Failure stops queue.

Prior work: FA-39 audit; FA-23 credit transfer (older single-head64KiBL2);
FA-01 code layout sensitivity; FA-06/29 distinguish direct P producer from existing
packing; FA-15–24 and FA-34/36/38 prevent duplicate generic load/copy/ring sweeps.
No accepted exact staging simplification or direct producer found in history.
Current user authorized implementing experiments to separate SPM-specific effects;
matched cache controls are included for that purpose, not a fresh cache tuning sweep.

Resource broker: one sim, two CPUs within existing24/128 affinity; 10GiB initial
lease contains8GiB simulator AS and2GiB coordinator/reference. Grows4GiB only within
shared20%RAM reservation/actual-use cap; >=64GiB host spare required. Stale host
monitor fails closed. Existing long-context/matrix runs remain untouched.

Run: python3 benchmarks/qwen/fa/bottleneck/run.py
Status: results/cacheflex_flash_attention/bottleneck_20261004/experiment_v1/status.json
Analysis: same root REPORT.txt and analysis.json, accepted cases only.
Initial prebuild string-count assertion attempts preserved; none launched simulation.
