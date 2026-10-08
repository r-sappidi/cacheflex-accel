# MICRO-derived Ruby timing profile

Implementation: `scripts/micro_timing_se.py`, used beside the frozen
`bounded_se.py` resource adapter. Enable with `--cacheflex-micro-timing`.
The default path remains unchanged; active unrelated experiments keep their
original inputs. Do not retrofit these timings into an in-progress run.

Reference: pristine MICRO revision
`82ac0b9de1ed408ce5cd53067267c8a855b4eef3`, especially
`experiments/end2end/scripts/sweep_vl16.sh`, `gem5/src/mem/cache/base.cc`,
and `gem5/src/mem/cache/tags/base_set_assoc.hh`. Its cache clocks use the
2.5GHz CPU domain, L1 tag/data2, L2 tag/data/response8, LLC35. Local crossbar
frontend/forward/return delays are1/1/1; the LLC crossbar uses10/10/10.

## Translation, not raw parameter substitution

Ruby's l0/l1/l2 controllers represent physical L1/privateL2/LLC.
The profile puts their service clocks in the CPU domain while leaving Garnet
router/link/NI clocks, topology and bandwidth unchanged. Memory size, ports,
miss/WB credits, sequencer guard, resource policies and kernel tiles are not
changed.

| Path | Ruby decomposition in CPU cycles |
|---|---|
| Actual L1 hit | Existing CPU/Ruby boundary stage + mandatory ingress1 |
| L1 miss/privateL2 hit | ingress1 + local forward2 + response11; response11 includes L2 service8 + local return1 + L1 refill2 |
| PrivateL2 miss | private controller request8, existing Garnet transport, LLC response35, private return/refill path |
| Pure SPM | existing CPU/Ruby boundary stage + ingress1 + forward1 + service/return5 |
| SPMCP | ingress1 + copy forward4, existing coherent source path, install acknowledgement8 |

The source-derived first profile double-counted one CPU-facing adapter cycle.
Same-ELF dependent-load probes exposed that offset; the final profile absorbs
it instead of adding it on top of MICRO's access latency. This calibration
uses isolated accesses, **not FA speedup**.

Different service delays feed the same ordered local FIFOs. The existing
`clamp_fifo_arrival` option preserves enqueue order by delaying a later message
whose nominal ready time precedes the previous message's ready time. It adds
neither entries nor bandwidth and does not let requests overtake each other.

## Verification and reproduction

Artifacts: `results/cacheflex_flash_attention/timing_alignment_20260922/`.

- `classic_build_v2`: rebuilt pristine Classic executable, revision/hash/config
  manifest and build commands. First SCons invocation compiled objects without
  emitting the final binary; the second emitted it. No upstream source changes.
- `guest_v4`: identical probe ELF for both models. Dependent randomized chains
  select L1, privateL2 or LLC residency; direct SPM uses64/128/256-byte predicates.
  Eight independent chains test L1/SPM throughput at256 bytes.
- `reference_v3`, `translated_v3`, `old_same_guest`, `streams_v3`: raw configs,
  commands, output and three statistics epochs. Only the final two warm epochs
  enter the calibration table. Prefetch is disabled for these probes only.
- `calibration.json`, `REPORT.md`: checked probe results and remaining differences.
- `fa_v3`: three ordering regressions, exact old-timing/export controls, full
  numerical T512 gates and a gated T4096 pair with the original fixed kernels.
  Prefetch remains enabled here. The numerical oracle/tolerance is unchanged.
  `verify_timing_fa.py --wait` performs an additional completed-output audit.

Useful commands:

```sh
python3 scripts/test_micro_timing.py
python3 scripts/analyze_timing_alignment.py
python3 scripts/verify_timing_fa.py
```

`run_timing_fa.py` creates a new immutable experiment directory; select a fresh
name with `CACHEFLEX_TIMING_FA_EXPERIMENT` when rerunning. Do not overwrite an
existing experiment. It freezes a copy of the timing/config adapter and checks
dependency hashes, exact outputs, arithmetic, copy counts and full configs.

## What this does not establish

This matches measured local hit latency and the tested independent-load
throughput. It is not complete Classic/Ruby equivalence. Ruby retains its
multicore mesh/directory rather than Classic's crossbars/snoop machinery.
The measured LLC dependent-chain residual is0.254 cycles (about0.28%). It was
not removed by fitting NoC parameters. Neither model gains a new SRAM bank
contention model here. WB-credit lifetime is unchanged. SPMCP timing is
source-derived but not calibrated across every source/ownership state;
SPMWB has correctness coverage, not exhaustive per-state timing equivalence.

MICRO sends a full-VL SPM packet even with a narrow predicate. All four ways
are initialized in those probes so inactive ways do not trigger its zero-data
fallback. Equal observed latency does not imply equal masked-access traffic.

The first FA candidate exposed a post-ROI functional-read filename problem.
The existing filename-refresh exporter repair was applied only to unused
legacy-main storage/post-ROI code. Default-timing controls reproduce all finite
non-host ROI counters and full outputs exactly. Earlier failed artifacts and
unmodified binaries remain available; no timed-kernel changes were made.
