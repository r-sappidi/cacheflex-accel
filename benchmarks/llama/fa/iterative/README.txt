FA-41: Llama-3.2-1B-shaped causal FlashAttention, 2026-10-04

Authorized: implement/run/review/iterate to defensible cache and private-SPM
baselines. Target assumed T4096 as in preceding Qwen/Llama discussion; batch1,
8cores,32queryheads,8KVheads,d64. Not complete model inference. Model architecture
reference: https://huggingface.co/amd/Llama-3.2-1B-Instruct-FP8-KV/blob/main/config.json
(the dimensional configuration of AMD's derivative; its quantization NOT used).
Qwen frozen matrix driver already admits32heads/8KV; exact guest and retained
simulator reused, no new hardware, L3commands forbidden. FP16 inputs and sequential
FP16 GEMM accumulation with FP32 online state as existing benchmark; not a claim
of standard BF16 model numerical equivalence. Full independent tiled oracle and
dense FP64 gates retain existing tolerances. Changed Bc gets separate reference;
exact output bits required across cache/SPM and geometries at sameT/Bc.

Prior evidence: FA35 V-only provisional winnerBr16Bc256R256; L3-11/12 cache
R256Bc256Br16; FA39 coherent load pressure, stage-address negative; MOD12 exact
work/capacity model useful but latency rank failed. FA30 is llama.cpp reference
with different precision/profile, NOT a Llama-model-shape baseline; cannot reuse
its timing. Inventory and current queues searched; no Llama32/8shape active.

Start cache/V Br16Bc256R256. Vpanel32KiB reused across16querymicrotiles. Four
query heads share eachKVhead, rather than seven. AggregateKV8MiB, Q/output16MiB
each atT4096FP16.512querytasks,64/core, rotated ownership balances causal work.
Do NOT assume full8MiB simultaneously live in privatecache. No alternative
placement matrix: V-only remains until measured evidence motivates redesign.

One simulator, shared20%CPU/RAM broker,10GiBlease(8sim+2coordinator),4GiB admitted
growth, fresh all-user host monitor,64GiBspare,2CPUs within existing24CPU mask.
No active source/guest/model/supervisor modified. No new build needed.

Iteration: each backend separately selects one unmeasured adjacent geometry
around lowest accepted full-ROI target. Counter evidence orders hypotheses:
coherent-load pressure -> smallerBc/R; bridgepressure and repeatedcopycount ->
largerR; instruction/Icache pressure -> Br batching; largerBc tests recurrence
amortization. These are hypotheses, not counter-to-time attribution. Analytical
before/after work/staging/footprints included in every decision. Actual target
latency selects next center; no extrapolated T256 winner. Both initial small
controls precede target. Each new geometry first passes all-head smallgate.
Bounds Br16/32,Bc128/256/512,R128/256/512, max8targetcases/backend. No Cartesian
product; no losing-placement tuning. Neighborhood exhaustion is LOCAL evidence;
budget exhaustion is unresolved. Final status awaiting_final_review, never an
automatic paper-ready/global-optimal assertion. Gains<0.5% treated as marginal;
small rank differences alone must not motivate architecture claims.

Decision criterion: substantial fullROI reduction with expected resource/work
change supports hypothesis; relief without gain demotes it. Software wins must
be available to bothbackends. Shared-pressure evidence does not justify SPM-only
architecture claim. Br64/other placements, code changes and model/hardware timing
are outside initial localloop and must be reconsidered only with evidence.
All accepted/null/failed results and decisions retained. Final review must inspect
counter trends, hardware/config equality, geometry boundary limits and whether
SPMpressure calls for a placement pivot before freezing baselines.
