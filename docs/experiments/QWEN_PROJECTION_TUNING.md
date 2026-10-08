# Qwen projection tuning (AI-11)

Related AI-01/04 explored older projection proxies and mapping; AI-03 showed
first-call packing can inflate benefits; AI-06/10 supply checked FP32-accumulating
GEMM primitives and the current Qwen finite hardware. This is a new Qwen-shape
transfer and bounded tuning, not a previously untested projection mechanism.
Decision: best accepted full first-invocation cache/SPM ROI at each token count,
including layout and bias, versus inherited MP2/KC128. Preserve regressions.

Qwen2.5-0.5B: fused QKV is [T,896] x [896,1152], widths896/128/128;
output projection is [T,896] x [896,896]. QKV has bias; output has none, as in
https://raw.githubusercontent.com/huggingface/transformers/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py
Synthetic FP16 inputs/weights/bias, sequential FP32 accumulation. This is not
BF16 checkpoint inference. Preconcatenated QKV weights are the common model-load
format for both arms. Cache first call includes its weight packing; SPM stages
original rows. QKV bias is added to preserved FP32 accumulators before FP16
rounding; the unchanged GEMM's earlier FP16 store is retained as overhead in
both arms. QKV exports Q14/K2/V2 each contiguous [heads,T,64]. Output projection
accepts FA's [14,T,64], and its timed ROI includes conversion to token-major.
No RoPE/norm/residual/logits included. Adapter copies must not be counted twice
in a later integration. Separate-vs-fused QKV scheduling is not swept here.

Original accepted AI-10 allocation/packing/staging/GEMM prefix and helper headers
are copied unchanged. New guest driver only. Same lifecycle build_v5 and full
hardware config as AI-10. 64KiB L1I+64KiB L1D/core,512KiB L2/core,4MiB LLC;
8cores,VL16,finite model. SPM eligibility6ways, panel uses4ways (32..224KiB).
No simulator changes, new L3 arithmetic, or modifications to active inputs.

Analytical model enumerates MP1/2/4/8 and KC128/256/512/1024. Exact per-core
arithmetic/copies/packing/partial-accumulator counts; 27 uncalibrated capacity/
movement sensitivities, scores not latency predictions. Independent equal
four-case budgets per operator/backend: inherited plus two ranked plus distinct
mapping diagnostic. Frozen selection.json records every rank and candidate.

Queue: T25 baseline tail gate for both operators/backends; all selected T24
operator/backend gates (including empty partitions); all16 candidates atT1024;
per-operator/backend first-call winner plus inherited atT2048/T4096. Full ordered
reference exact bits, separate float64 oracle, all-output hash, exact peer bits,
copy counts, no reductions, LSQ/global bounds, complete config comparison.
Two measured invocations; second is a reuse diagnostic, not steady-state median.
A failure stops downstream with evidence preserved. No tolerance relaxation.

At most1 new simulator,6total. Existing jobs untouched. Launch idle>=50%,
RAM>=128GiB,disk>=100GiB; runtime RAM>=64GiB,disk>=75GiB;96h per-simulation.
One-thread reference generation. Queue may wait for slots; status.json explicit.
Root results/ai_workloads/qwen_projection_20260928/experiment_v1.
