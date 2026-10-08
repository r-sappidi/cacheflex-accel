# Bounded Ruby flight recorder

For newly built MESI_Three_Level_SPM simulators, append this **configuration**
option (after `se.py`) to a future run:

```
--ruby-flight-recorder-entries=8192
```

Default is disabled. This applies to all instantiated Ruby controllers,
including private caches, shared LLC banks and directories. Per-controller
overrides are also possible via `-P system.ruby.l1_cntrl0.flight_recorder_entries=8192`.
Old frozen simulators do not support the new parameters. Existing running jobs
are unaffected; do not replace their binary or restart them to enable this.

Each controller creates `<controller-name>.flight.bin` in the gem5 output
directory. Each file is exactly `64 + N*1024` bytes. At N=8192, 32 controllers
use approximately 256 MiB total, independent of run length. Existing recordings
are never silently overwritten: use a fresh output directory for each run.

N counts **diagnostic records**, not completed memory transactions. Records
include tick, controller (filename), sequence number, transition event/address,
old/next state, outcome (including stalls), and message SEND/PEEK observations
with queue and the message's standard printed fields. PEEK can repeat and does
not prove consumption. These are controller observations, not all router hops.
Long messages are marked `[TRUNCATED]`. Transition begin records preserve context
if an action crashes before returning. Optional
`--ruby-flight-recorder-transitions-only` reduces host overhead and record volume.

Files are Linux shared memory mappings, written as events happen; no exit handler
is needed to recover them after panic, abort, or SIGKILL. This does not guarantee
survival of a host power failure. Sequence tags and checksums reject interrupted
overwrites. The single-writer recorder does not schedule simulated events or
generate simulated traffic, but formatting/writing can increase host runtime.

Decode after a crash, or take a best-effort live snapshot:

```
python3 scripts/decode_ruby_flight.py RUN_DIRECTORY > recent-transactions.log
python3 scripts/decode_ruby_flight.py RUN_DIRECTORY --contains fc9040
python3 scripts/test_ruby_flight.py
```

The decoder merges retained records by simulated tick and controller/sequence.
Same-tick events across different controllers do not have a global ordering.
High-volume retry activity can overwrite the original initiating request; choose
a larger N when needed. A bounded ring is context, not a guarantee that every
dependency involved in a deadlock remains available.

## Validation

Built simulator: `results/ruby_flight_20260916/build_v2/gem5.opt`.
`results/ruby_flight_20260916/validation/passed.json` records the T512 Q-resident
FA off/on gate: independent numerical checks, matching output hashes, matching
extracted ROI counters and exactly 113.8272 us simulated ROI in both runs.
All 32 controller files decoded successfully. Host time was 22.95 s disabled
versus 44.74 s enabled in this check; recording is not free in wall-clock time.
Standalone tests also verify SIGKILL recovery, bounded wraparound, truncated
messages, rejection of torn/corrupt slots and refusal to overwrite existing files.
