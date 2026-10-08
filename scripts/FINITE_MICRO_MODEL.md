# Finite MICRO-derived Ruby profile (Sep 26)

This is an isolated corrected-model experiment, not a performance optimization
claim. Existing unlimited-transport tuning results are preserved and must not
be combined with new timings into a matched speedup or tuning fit.

## Resources

The prior MICRO-derived profile limits actual misses and writebacks, but leaves
bridge shadow tracking and most Ruby transport queues unbounded. A Bc=1024
query-SPM gate reached the 1024-request sequencer guard. Increasing that guard
alone would not establish a finite hardware model.

The new opt-in profile uses:

- Regular and SPM LQ32/SQ48. Pending bridge shadows share the existing SPM SQ
  budget, and detached accepted loads retain credits until response.
- Existing MICRO-derived credits: separate L1I/D 8 misses each, private L2
  20 misses, pooled LLC 32; writeback credits 8/8/16.
- L0/private-L2/LLC/directory TBEs 16/32/64/64, per-line targets 16.
- Each Ruby MessageBuffer 16 entries, except the ordered local regular request
  lane (12) and recall/eviction return lane (4). Pure-SPM ingress has its own
  existing direct path with 16 entries. No new SPM data SRAM.
- Sequencer 1024 solely as a safety guard; exact 4096-counter equivalence and
  zero cap rejects required before reporting it as nonbinding for tested cases.

Queue and TBE sizes are explicit modeling assumptions, not silicon-calibrated
values or a claim of exact Classic implementation equivalence. The stress
profile uses LQ/SQ8, pools2, TBEs8, targets4, queues4, local request/return2+2.

### Sep26 stress repair: preserve four request VCs, split by request stage

The first profile passed all nominal gates and exact1024/4096 equivalence, but
both stress cases deadlocked before ROI. LLC-bound and directory-bound requests
used the same request VCs. LLC input backpressure could prevent the directory
fetches needed to release the LLC's occupied miss credits. See
`finite_model_20260926/diagnosis/REPORT.md` and preserved trace windows.

The opt-in `cacheflex_request_vc_partition` assigns the lower half of the four
existing request VCs to LLC-bound traffic and the upper half to directory-bound
traffic, at injection and every hop. Ordered-vnet checks also respect the split.
Total VCs, VC depths, network bandwidth, cache capacity, and all controller
resource counts remain unchanged. Both cache and SPM use the split. Defaults
leave the previous routing/allocation behavior unchanged. This is an explicit
finite-network design assumption and forward-progress repair, not an SPM-only
performance optimization. Stress acceptance is required, not presumed.

## Integration repair

The old finite-buffer profile reused `spmBufferFromL0` for recall responses.
The newer MICRO profile uses that input for direct SPM accesses. Combining
them would place recalls behind potentially blocked requests again. A distinct
ordered `returnBufferFromL0` input now carries both PUTX and INV_DATA/INV_ACK,
and is serviced before either request input. Existing escape transitions keep
their state/data actions, reading and popping the new return input. Default-off
mode sends replies through its original shared request lane.

New binaries require `finite_micro_se.py` (frozen as `bounded_se.py`) to bind
this input. Active old binaries/configs were not edited. No latency, NoC width,
FU, cache geometry, replacement, retention, or kernel changes are intended.

SPMWB still uses the existing conservative shadow reservation in SPM SQ, not
the paper's exact placeholder-LQ design. Roundtrip/RAW/GETX regression tests
cover exercised behavior, not exhaustive coherence proof. DMA, multicast,
batch and full-system operation are outside this validation scope.

## Acceptance sequence

`run_finite_micro.py` runs at most two additional simulators:

1. Default-off T1024 cache/SPM: all finite non-host counters and outputs equal
   prior accepted runs (only statistics for the new unused return buffer omitted).
2. Bridge roundtrip, RAW without post-WB barrier, cross-core release/GETX tests.
3. Eight T1024 numerical/configuration gates covering query and KV partitioning,
   cache/SPM, and the formerly failing Bc1024 case; full output checks, exact
   copy counts, zero guard rejects, observed LQ/SQ/copy bounds.
4. Repeat all eight at guard4096: exact numerical non-host counter equality.
5. Two small-resource stress cases; failures stop downstream work.
6. Matched four-family T4096 and T16384 fixed candidates under the new profile.
   These are pilots for subsequent re-tuning, not new best-to-best minima.

The repaired queue runs stress immediately after bridge regressions so the
known failure is checked before nominal/guard repeats. `run_finite_recovery.py`
waits for its isolated build, runs validation and pilots, then starts
`run_finite_analytical_tuning.py`. The latter seeds the existing equal-budget
model-guided search with ONLY eight matched finite-profile pilots. It excludes
those seed configurations from repeat candidate selection, preserves all input
hashes, performs numerical/configuration gates, then model-ranked plus exploratory
T4096/T16384 candidates and local refinement. Any prerequisite failure stops
the chain; historical unlimited-transport timings are not training samples.

Results and source preservation: `results/cacheflex_flash_attention/finite_model_20260926`.
`status.json` is authoritative. A built binary is not a validated model.
