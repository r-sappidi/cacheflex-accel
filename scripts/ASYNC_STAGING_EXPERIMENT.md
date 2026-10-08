# Bounded copy isolation and early staging

Question: can reducing unrelated-load backpressure and bringing the next V tile
in early improve whole-FA latency without adding data SRAM or memory bandwidth?

## Controls

All experiments use eight cores, VL16, one attention head, d64, 64KiB private L2,
4MiB shared LLC, 128-bit network links, and 16 total sequencer credits per core.
SPM uses LLC retention. Cache remains the existing independently tuned
Br32/Bc128/G2 implementation; original SPM is Br16/Bc128/G4.

Hardware options are default-disabled:

- `cacheFlexIndependentBridge`: bridge requests use a distinct core request port
  and retry flag. SPM retries also no longer clear or set coherent backpressure.
- `cacheflex_max_bridge_requests=12`: at most 12 outstanding bridge requests
  within the existing total of 16. Ordinary traffic can still use all 16 when
  copies are absent. This is not an additional credit pool.

Issue bandwidth, dual-LSQ dependency rules, coherent protocol, controller/network
resources and latencies remain unchanged. Bridge instructions still use the
ordered regular SQ, so this does not eliminate head-of-line blocking of later
stores. The mechanism is narrower than a dedicated asynchronous DMA engine.

## Software modes

A separate guest supports three modes with one executable:

1. JIT: alternate two buffers, staging each tile immediately before using it.
2. Panel-ahead: issue copies for the next panel before current-panel computation.
3. Chunk-ahead: distribute next-panel copies over the current query-group work.

Each V tile occupies 16KiB. Double buffering therefore uses four existing ways
instead of two, reducing coherent capacity. The four-way JIT control measures
this storage tradeoff independently of early staging. No new SRAM is allocated.
Physical buffer addresses use ways 0/1 and 2/3, sets 0..127. Original arithmetic
sequence and total copy count are preserved; correctness is checked independently.

## Queue and acceptance

`run_async_staging.py` runs 12 T512 correctness/default-equivalence gates, then
the same 12 cases at T4096: cache, cache-isolated, original SPM, cap-only,
retry-only, both, and each of the three software modes with/without isolation.
It selects the faster combined early-staging variant at T4096, then runs six
T16384 cases: cache, original SPM, JIT shared/isolated, selected mode shared/isolated.
This is a targeted experiment, not exhaustive tuning or proof of a global optimum.

Every accepted case has full-output independent reference validation, export/hash
verification, exact copy counts and a full configuration audit. Default T512
cache/SPM must match earlier accepted extracted counters and output exactly.
Failures stop the queue and peer simulations. At most two new simulations run.

Results are under
`results/cacheflex_flash_attention/async_staging_20260921/experiment`.
`status.json` is authoritative; `REPORT.md` reports measured whole-ROI times.
Primary comparisons are new vs original SPM and new vs cache on matched hardware.
Do not add individual counter reductions together as causal runtime savings.

The preserved source tree contains pre-change versions; original frozen simulator
and kernel artifacts remain available. The final experiment binary is
`build_v2/gem5.opt`; the earlier `build/gem5.opt` lacks the bridge cap.
