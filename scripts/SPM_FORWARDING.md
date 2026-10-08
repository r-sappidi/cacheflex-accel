# Existing-SQ forwarding for SPM loads

`cacheFlexSpmStoreForwarding` is opt-in (default false). A pure SPM load
may consume the payload of the youngest older overlapping pure SPM store
already held in the finite SPM SQ. No dedicated SPM backing or new data
array is added. The forwarding comparator/multiplexer and wiring would be
additional hardware; their area and timing have not been synthesized.

The candidate must cover the entire load, have resolved address/data, and
not be masked, atomic, or completed. Any unresolved older store disables
forwarding conservatively. Partial coverage falls back to the existing
acknowledgment-based wait path; there is no multi-store merging. Existing
bridge RAW/WAR checks remain ahead of forwarding. The address comparison
uses slot plus way/byte ranges, not contiguous encoded-address arithmetic.

A snapshot enters the existing load result buffer and the standard O3
writeback event. Forwarding takes two modeled CPU cycles and consumes the
existing SPM load-port budget, charged per 64 bytes. Two cycles is an
explicit design assumption, not measured circuit timing. The O3 scheduling,
queue capacities, store commit/installation and Ruby timings are unchanged.

## Experiment

Build: `python3 scripts/build_spm_forwarding.py` (pre-edit sources already
preserved; do not repeat `--preserve`). Frozen simulator and configs reside
in `results/cacheflex_flash_attention/spm_forwarding_20260924/build`.

Runner: `scripts/run_spm_forwarding.py`; at most two simulations. Check
`experiment_v3/status.json` before starting anything. Stages:

1. Host address-range checks; guest OFF/ON back-to-back stores, youngest
   writer, different slots and partial-overlap fallback (16,384 elements).
2. Existing bridge/bootstrap plus wide-store roundtrip, OFF/ON (4,096 bits
   patterns across 64 rounds).
3. Identical output-resident FA guest OFF/ON, T512 numerical/equivalence gate.
4. Identical output-resident FA guest OFF/ON, T4096 target measurement.

Eight cores, single head, d64, VL16, Br16/Bc128/G4, 64KiB private L2 and
4MiB shared LLC. The FP32 output panel uses two existing SPM ways. No kernel
or memory-capacity change. All outputs are compared to an independent
arithmetic reference and the prior bitwise output; configuration, copy,
store and release counts are checked. OFF must exactly reproduce all finite
non-host counters of the previous repaired model, allowing only the new
zero-valued forwarding-byte counters.

Report OFF/ON end-to-end ROI speedup separately from comparison against
the existing optimized cache winner. Load retries are diagnostic events,
not additive lost-runtime cycles. T512 is not a substitute for T4096.

Initial regression attempt used the wrong scalar mnemonic size (64 rather
than 8 bytes), which encodes a release; it failed OFF and ON. Failed guest,
source, stats and logs are preserved in `tests`, `diagnostic`, `experiment`.
The corrected test uses `SPMSTR_8_IMM`; no hardware change was made in
response to that test failure.

T512 subsequently passed all arithmetic, output and OFF-equivalence checks:
OFF and ON are both 172.2084 us with zero forwards and identical counters.
The runner initially stopped on an inappropriate requirement that FA itself
must forward. The positive exercise requirement belongs to the focused
regression (259 correct forwards), not the workload outcome. V3 retains those
validated small results and runs the T4096 pair without changing the model.
See `spm_forwarding_20260924/analysis/REPORT.md` for an observed unresolved
older-store address blocking a load to a different slot. This is a separate
dependency from the one forwarding addresses.
