# Bounded stale-owner silent-fetch retry

Validation update: all14 simulator gates passed in experiment_retry_v2. Original
deadlock replay216.726us,65536 independent outputs pass,51NACKs and51reissues.
Config comparison against the actual failed run changes only new retry flags
and delays; every old field is identical. Both small-resource stress tests and
exact observer-on/off statistics/output equivalence pass. Matched T4096 targets
are running, T16K queued. See the experiment's REPORT.md and status.json.

Related evidence: FA-31 (placement gate failure), MOD-03 (finite resource/network
model), FA-29 (current tuned controls). This is a protocol progress repair and
placement revalidation, not a new throughput mechanism or added SPM storage.

## Failure

The preserved full-set T1024 / Br64 / Bc1024 / R256 / stride512 case deadlocks.
Its bounded trace reconstruction finds all 16 entries occupied in core0's
requestToL2, all 16 in requestFromL2, and all 16 in LLC7's forwarding output.
The forwarded GETS_SILENT for 0x1013c0 reaches core0 in I. The old transition
requires a slot in requestToL2 to bounce to the home before consuming the
forward. That backwards request-channel dependency is unsafe under finite
backpressure. FIFO contents and packet wait evidence are retained in
`spm_set_spread_20260927/diagnosis/fifo_evidence.json`.

## Repair

Opt-in `L1Cache.spm_silent_retry` replaces ONLY stale-owner bounces (I, IS, IM,
IS_I, IX_A) with a response-channel SPM_SILENT_RETRY addressed to the original
requester. It carries source line and destination SPM slot identity, not data.

The requester consumes this response immediately into its existing IX_D TBE.
It keeps the original miss credit, destination, and ordering obligations. After
eight controller cycles it retries GETS_SILENT through the ordinary bounded
request FIFO. The timer can wait for FIFO space without occupying an incoming
response or forwarded-request slot. No line is installed or request completed
on a NACK, and no coherent owner is fabricated.

One retry marker is allowed per live TBE; assertions check live transaction,
matching destination, no duplicate pending retry, and no TBE deallocation while
a retry is pending. TimerTable is simulator scheduling for this bounded retry
state, not additional transaction capacity. The retained access-mode metadata
and retry scheduling need small control state; they do not add a data array,
miss credit, queue slot, VC, port, or bandwidth. No infinite retry queue is used.

Generated SLICC transitions confirm NACK consumption has no output-resource
check, while retry issue still checks the existing request FIFO. This removes
the observed forward-to-request edge; it is not an exhaustive liveness proof
for every protocol state or untested DMA/multicast mode.

## Validation and continuation

Isolated build_retry_v3; v1/v2 are compile-only failures, not simulator results.
Original failed simulator and trace remain untouched. `run_spm_retry.py`:

1. Default-disabled cache/SPM equivalence: every original finite numeric nonhost
   statistic and full output bits; only new zero event counters are permitted.
2. Bridge roundtrip, unfenced RAW, release/GETX, and wide four-way tests.
3. Exact original deadlock case, with nonzero NACK/reissue counts required and
   complete independent arithmetic validation.
4. Repaired cache/SPM controls, remapped small case, cache and SPM tiny-resource
   stress, and exact observer-on/off equivalence.
5. Only after gates pass: matched repaired cache/control-SPM/spread-SPM at T4096,
   then T16384. Existing guest and all storage/resource budgets remain fixed.

Primary performance endpoint is full FA ROI latency. Repair effects must not be
confounded with placement effects by comparing a repaired SPM run against an
unrepaired cache control. This run does not claim bank-calibrated timing or
global kernel optimality. Failures halt this queue; unrelated llama run is left
alone. Maximum two new simulations at a time.
