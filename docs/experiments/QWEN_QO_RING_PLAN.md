# FA-33 / FA-34: resident Q/output followed by neighbor-ring evaluation

User authorized 2026-09-30: first start combined Q+accumulator staging, then
execute the staged ring evaluation. No earlier Qwen combined-residency result
was found. Related FA-10 tested Q alone on older 64KiB L2 and exposed a six-
instruction broadcast adapter; FA-25–27 tested private output residency and
forwarding/address changes; those did not beat the strongest baseline. FA-14
double buffering regressed on the older profile. FA-28/29 tested query versus
KV partition, while GEMM-05 multicast cut copies with only ~2% incremental gain.
L3-06/07/08/11/12 supply accepted current Qwen cache/V-only references.

## FA-33: new combined layout, unchanged hardware

Batch1 causal FP16 Qwen attention, Hq14/Hkv2/d64, eight cores; Br16/Bc256.
Process heads sequentially. Each core owns two complementary contiguous query
blocks: block c and block 15-c out of 16 equal blocks. At T4096 this gives
512 rows/core (256 early +256 late), balancing triangular causal work while
retaining the entire local head slice. Each core keeps its own output; no merge.
Future-only Br16/Bc tiles are skipped. Per-head barrier included in ROI.

Same new ELF, placement factorial: coherent; rolling V; resident Q+FP32 O;
resident Q+O plus rolling V. Two legacy-schedule modes supply same-ELF current
cache/V controls. All arithmetic remains CPU, no shared-L3 SPM or reductions.
Q uses way4, up to1024 slots (64KiB); O uses ways0–3 sets0–511 (128KiB);
V uses ways0–1 sets512–767 (32KiB). These are disjoint slots in existing SRAM.
Private stores do not implicitly claim slots: O is initialized by four real 64B SPMCPs per row from coherent zero scratch, followed by a timed drain. Q uses two 64B copies per row. All stages/releases and CPU normalization/export stores are timed. Scalar
softmax state and temporary score/probability panels stay coherent. Q adapter
preserves existing FP16 FMA order and uses established 64B SPMLD plus TBL;
O uses direct SPM loads/stores in the paired PV epilogue, not buffered PV output.

Gates: static ownership/layout/arithmetic audits; small full-output factorial;
T1024 full factorial; T4096 full factorial and fresh legacy controls. Independent
full oracle, exact same-Bc output bits, first-row identity, exact copy counts,
config-delta/resource bounds, zero reduction commands. Fail closed on any gate.
Do not call compiled code or small gates a performance result. No tolerances
relaxed. Target decision: complete FA ROI versus identical-layout coherent/V
and the strongest accepted existing private9473.3184/cache10509.4604us.

## FA-34: authorized downstream ring evaluation, gated on FA-33

First evaluate optimistic neighbor delivery on the validated layout: finite
existing SPM buffers, actual initial staging/finalization/CPU arithmetic, but
explicitly optimistic transfer timing. Compare against the identical layout
with ordinary staging; ideal-only timing is a headroom diagnostic, not a design.
If there is no useful full-ROI headroom versus strongest private, stop finite
ring performance work and report the negative decision.

Then finite neighbor copies with bounded credits/completions, same physical
SRAM, explicit read/write port service, link serialization and backpressure.
Directed gates cover one hop, eight-way simultaneous ring, wraparound, buffer
reuse/RAW, stalled receiver, minimum resources and recovery. Initial coherent
claims/source fetching retained; transfer only between already-owned SPM slots.
No arbitrary remote loads, no unbounded payload queues, no free SRAM ports.
Matched ring/no-ring kernel tile order and independent arithmetic validation.
Conservative width/latency/credit sensitivities, then another FA size and a
GEMM workload only after a positive realistic result. Full end-to-end ROI must
beat same-layout staging AND existing tuned controls; traffic/stall sums alone
do not advance the decision. Ring attention is established algorithmic prior
art (https://arxiv.org/abs/2310.01889); the experiment concerns on-chip SPM
transport/ownership, not novelty of circulating K/V.

## Execution isolation

Root results/cacheflex_flash_attention/qwen_qo_ring_20260930. Frozen old source,
ELFs/configs/results unchanged. New guest only for FA-33; reuse lifecycle
build_v5. Shared OPS-23 reservation broker, <=2 new simulators, same24/128 CPUs,
all-user CPU/RSS cap20%, >=64GiB host headroom beyond remaining allowance.
New jobs wait for broker capacity; do not resume old stopped coordinators.
FA-34 is a dependent work stage, not an implemented/queued simulator claim
until its model and executable pass the recorded gates.
