# Qwen-shaped FA at 16K and 32K (L3-10)

User authorized queueing T16384 and T32768. Related FA-28/29/31 already
studied older single-head T16K mappings/placement: less traffic did not guarantee
a best-to-best win. L3-06/07/08 established/tuned Qwen Hq14/Hkv2 causal FA through
T4096; L3-09 adds finite direct links and finds small incremental transport gains
at T1024/T2048. Inventory has no accepted Qwen T16K/T32K full-head counterpart;
old single-head records mentioning Qwen templates are not long-context cases.

Materially new: transfer the accepted Qwen configurations to larger supported
context lengths, testing capacity/sequence scaling for cache, private SPM,
NoC L3 reduction, and direct L3 reduction. Decision: full-ROI speedup over cache,
over private SPM, and matched direct-vs-NoC delta. Report regressions/plateaus;
these inherited tiles are not independently tuned long-context optima.

## Queue and controls

Wait for direct_link_20260928/experiment_v2 to complete successfully. Use its
frozen build_v4, configs and best T4096 per-backend choices: Br16/R256 throughout,
cache Bc512 and private/L3 Bc256. Same geometry for NoC/direct L3. Four new-guest
T1024 controls must exactly match their accepted old-guest output and every
finite nonhost statistic. Then four arms at T16384 followed by four at T32768.
No additional tile sweep, simulator changes, or changes to existing queues.

The old guest rejects T>16384. Isolated guest_v2 changes only that guard to32768;
binary .text audit finds exactly one MOV immediate changed at0x400614
(16383 ->32767, unsigned t-1 guard); all other .text and headers unchanged.
Guest_v1 compiled, but a host audit incorrectly expected a CMP immediate;
its failed audit is retained. Guest_v2 uses the actual compiler encoding.
Causal u16 indices max32767; array counts fit int; guest memory remains2GiB.
Use the same new guest for every arm at both lengths, after exact controls.

Full independent sequential-FP16-FMA/FP32-online reference for every output,
separate reference per Bc; unchanged existing tolerance and exact same-Bc peer
outputs/export hashes. Reference computes64-row chunks, not a T-by-T dense
score matrix, in one host process with BLAS/OMP threads=1. No sampling shortcut.
Generated references have SHA manifests. Guest per-core work, copy/reduction/
init/complete/transport byte and beat counts are derived from logical work.
Configuration audit allows only guest executable and token-count argument plus
known direct-link flags. Same bounded LSQs/queues, SRAM/ALU timing and geometry.
T16K exports14680064 elements; T32K29360128. All must pass before acceptance.

At most one new simulator and six total; >=50% CPU idle, >=128GiB available
memory, >=100GiB disk at launch; >=64GiB memory/>=75GiB disk while running.
Single-thread reference jobs obey the same headroom check. Allow96h per long
simulation/reference job to avoid the earlier12h cutoff for quadratic scaling.
A timeout or arithmetic/configuration failure stops this queue and preserves
all partial evidence; it never relaxes acceptance or alters other jobs.

## Physical caches (unchanged)

Per core:64KiB4-way L1I +64KiB4-way L1D;512KiB8-way private L2.
Shared L3:8 banks x512KiB16-way =4MiB total.64-byte cache lines;8 cores.
Ruby labels l0/l1/l2 correspond to physical L1/L2/L3 respectively.
SPM repurposes existing SRAM, no separate backing array. Six private L2 ways
are eligible (up to384KiB/core); eight shared L3 ways eligible (up to256KiB/bank).
Eligibility is not a fixed reservation: actual claimed slots consume capacity.
Inherited Bc256 private V panel is32KiB/core; R256 FP32 L3 O is64KiB/worker bank,
512KiB across8banks. Q/K, softmax and other coherent data still use caches.
Cache-only arms have no SPM claims. FP16 K+V is8MiB at16K and16MiB at32K.
Model shape is Qwen2.5-0.5B-Instruct, B1/Hq14/Hkv2/d64, causal, FP16 GEMMs and
FP32 online state; this is not full-model BF16 inference or an energy estimate.

Authoritative queue: results/cacheflex_l3_spm/qwen_long_context_20260928/experiment_v1.
