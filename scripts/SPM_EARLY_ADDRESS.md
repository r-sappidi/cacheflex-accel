# SPM store address/data separation

Motivation: the T512 trace shows load SN1551468 (slot 2) waiting 245 local
cycles until older stores SN1551394/1551395 resolve to disjoint slots 0/1.
The load issues before those stores complete. Store-to-load forwarding
therefore cannot remove this particular wait. Its T512 and T4096 FA tests
both measured exactly 1.000000x, with zero FA forwards.

## Model

`cacheFlexSpmEarlyAddress` defaults OFF. The experiment adds one dedicated
address-only unit per core, issuing at most one computation/cycle with one
cycle latency. This is explicit added address-generation/control hardware,
not a free bypass or additional SPM data capacity. Area and circuit timing
are not synthesized.

When a younger SPM load encounters an unresolved older store, the unit may
read that store's **ready renamed integer base operand**, calculate the
physical way/slot range, and record it in the existing finite SQ entry.
Only the SVE immediate-address SPM store exposes the decoder hook. Scalar
stores, bridge instructions and unready bases retain the old behavior.
The initial experiment uses fixed VL16 in syscall emulation.

The load may proceed only after the early result becomes available and
proves disjoint. Same-slot overlapping loads still wait for the store.
This path neither executes the store nor makes its data available, commits
it, translates an address, accesses memory, or bypasses bridge ordering.
Metadata is cleared with the SQ entry on retirement/squash. Actual store
execution asserts that the early address and full extent were exact.
Predicate masking is conservatively represented by the full possible range.

Store-to-load forwarding stays OFF in both experimental cases, isolating
address resolution. The kernel, capacities and memory latency are unchanged.

## Reproducibility

Build: `scripts/build_spm_early_address.py`, frozen `build_v4`.
Runner: `scripts/run_spm_early_address.py`, `experiment_v4/status.json` under
`results/cacheflex_flash_attention/spm_early_address_20260924`.
Check status before launching duplicates. Maximum two simulations.

Gates cover delayed store data with disjoint and overlapping loads, delayed
bases, branch-dependent writes, predicate and immediate-offset cases, prior
partial-overlap/youngest-writer tests and bridge/wide-store roundtrips.
Then T512 and T4096 OFF/ON pairs use the unchanged output-resident FA guest:
eight cores, single head, d64, VL16, Br16/Bc128/G4, 64KiB private L2,
4MiB shared LLC and two existing SPM ways.

Every FA case requires independent arithmetic, bitwise output equality,
configuration/copy/store/release audits. OFF must reproduce all prior finite
numeric non-host counters exactly, allowing only 64 new zero counters.
Report end-to-end ROI speedup and comparison to the existing cache winner;
disjoint-check counts are not additive saved cycles.

The initial generic decoder-template hook failed compilation because that
template also serves register-offset instructions. The v2 hook is confined
to the dedicated SPM store declaration. Failed build logs are preserved;
no simulator from that attempt was used for measurements.

V2 compiled and exercised early addressing, but the regression found 1,536
inactive predicate lanes overwritten. The frozen pre-feature simulator gives
the same 1,536 errors: this was an existing SPM store-mask defect, not a reason
to weaken the test. V3 carries the request byte enables through Ruby and uses
them as SRAM write enables. Fully active writes retain the original fast path;
mask repair is present in both arms. V3 stopped during SLICC generation before
producing a simulator; V4 corrects the message-field declaration location.
FA's unmasked OFF counters must still
match exactly. Old-model and V2 failed evidence remains preserved.

`tests_original/guest` matches the original failing guest's manifest SHA256
exactly. The current regression additionally delays the immediate-offset
store's data so early address generation is exercised on that form too.

## Accepted initial result

V4 passes all six tiny OFF/ON cases and the full T512 numerical/bitwise/
configuration gates. OFF reproduces prior finite non-host statistics exactly.
T512 OFF is 172.2084 us; ON is 171.8204 us: **1.002258x**, or **0.2253%**
less end-to-end latency. ON performs 3,356 early address computations.
SPM load rescheduling attempts fall from 178,844 to 51,492. That reduction
does not translate proportionally to runtime and must not be presented as
saved critical-path cycles. The matched T4096 pair is queued automatically
after these accepted gates; its status/result is in experiment_v4.
