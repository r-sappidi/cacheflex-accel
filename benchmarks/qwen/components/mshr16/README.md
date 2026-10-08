# AI-13: Qwen FFN/projections at 16 L1D miss credits

Question: do the AI-10/AI-11 cache-vs-SPM speedups (FFN 1.10-1.36x, QKV/output
about 1.30x first call) survive a cache baseline with 16 outstanding L1D line
misses per core, tuned over tilings that fit L1?

Why prior evidence is insufficient: AI-10/AI-11 ran at 8 L1D miss credits.
FA-39 showed 8 -> 16 credits shrinks the FA SPM lead from 1.110x to 1.015x
at T4096. The AI-10 T4096 cache winner (MP8/KC512) kept all 8 credits busy, with
22.4M L1D misses vs 1.87M for SPM. The cache candidates never included an
L1-fitting balanced tiling (MP8/KC128 or KC256) and never KC1024. SYN-02 showed
that the fastest tiling can differ from the most energy-efficient one.

## Design

- Platform: frozen AI-10/AI-11 simulator (lifecycle build_v5), configuration
  script, guests and inputs. The only added override is
  `system.ruby.l0_cntrl{0..7}.dataMissCredits.capacity=16`. Unchanged: 16 L0
  TBEs, 16 targets per line, 20 private-L2 credits, VL2048.
- Replay gate: two accepted AI-11 T1024 runs at 8 credits must reproduce the
  same command (except outdir), timings and output bits exactly.
- Candidates: MP8 x KC {128, 256, 512, 1024} for both backends. The B panel per
  128-column tile is 32/64/128/256 KiB, so KC128 fits in the 64 KiB L1 and KC1024
  minimizes accumulator traffic (FFN T1024: 793 vs 73 MB). This is a bounded axis
  between those two costs, not a Cartesian sweep. MP8 won every AI-10/AI-11
  screen, and combined gate-up lost in AI-10.
- Stages per workload (QKV, output, FFN), with pipelines sharing four simulator
  slots: T24 gates for all 8 candidates; then a T1024 screen of all 8; then an
  8-credit attribution of each cache tiling not run in AI-10/11 (8 runs total).
  Each backend's T1024 winner by first-call and by repeat-call ROI (union, at
  most 2) transfers to T2048 and then T4096.
- Acceptance: unchanged AI-10/AI-11 checks (guest result lines, export-bit
  hash, full reference/ordered math audit, copy counts, LSQ bounds, zero
  reduction, global guards). Additional checks: config delta limited to the
  credit capacity, which must be uniform; maxLive <= credits; output bits equal
  to accepted AI-10/AI-11 bits for the same workload and T; exact peer bits.
- Resources: shared broker with 10 GiB leases, an 8 GiB initial address-space
  limit with admitted +4 GiB growth, a pause on pressure, the 20% CPU/RAM
  budget and 64 GiB headroom. Budget CPU offsets 4-11 (2 per simulator); the
  coordinator uses offsets 2-3; FA-41 keeps 0-1. Watchdogs: 900 s without
  committed-instruction progress, 96 h wall time.
- Analysis (`analyze.py`, reads accepted runs only): best backend per (workload,
  T, credits) for first and repeat call; SYN-02 energy per call. Includes 8-credit
  AI-10/11 rows, with differing candidate counts shown.

## Decision

If the best-vs-best SPM lead at 16 credits stays at or above about 1.10x
(repeat call), GEMM/FFN carries a credible multicore speedup claim. If it drops
to about 1.0-1.05x, the speedup story is the same as FA and the claim shifts to
energy (with the tiling-matched energy check).

## Limits

- Bounded MP8 KC axis only; no global-optimum claim.
- VL2048 and the old protocol build are kept for comparability with AI-10/11,
  while FA uses the retained-protocol build.
- The repeat call is a warm diagnostic, not prepacked deployment.
- Synthetic FP16/FP32 operators, not full-model inference.
