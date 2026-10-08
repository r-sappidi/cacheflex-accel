# Repository state — 2026-10-07

Operational snapshot after the October 7 reboot; see [OPS-26 in the ledger](LEDGER.md).
Refresh `python3 scripts/refresh_current_result_log.py` and inspect
`python3 benchmarks/status.py` before using this snapshot for future work.

## Subsequent update — FA-53 completed October 7

The standalone monitor was restored for the authorized exact-winner profiling replay.
FA-53 completed with full oracle/phase/frozen-input checks and exact FA-51c output
bits and all non-host statistics. Full ROI 8398.4216 us; critical core 2.
Softmax 38.487%, PV 24.717%, QK 19.863%, P packing 5.655%, V staging 3.879%,
output accumulation/rescaling 2.580%. See
[validated profile](../../results/cacheflex_flash_attention/winner_profile_20261007/REPORT.md).
Registry now has 635 accepted records, 71 queues, zero discrepancies. The
FA-53 lease was released; no old queue was restarted. The earlier snapshot below
records the state before this authorized monitor restart/replay.

## Runtime and resource state

Host boot: **2026-10-07 11:47:17 PDT**. No gem5 simulator, experiment coordinator,
build process or resource monitor was present in the account process inspection.
AI-13 (70 accepted), AI-13b (11 accepted), and FA-51b (3 accepted) still have historical
active-job entries; those PIDs are absent and their records predate this boot.
FA-51b was partly superseded by FA-51c; dropped flag-13 work is not pending authorization.
FA-41's status still says `awaiting_final_review`, but its October 6 final review
is recorded in the ledger and `results/ordered_fa_followup_20261005/llama/FINAL_REVIEW.md`.
No experiment or monitor was restarted by this cleanup. Coordinator-owned status
files were preserved, not relabeled as if coordinators had updated them.

The broker contains zero leases, a retained 6 GiB `old_reserved` allowance, and
`hold_new=true` with a stale heartbeat. Its last monitor sample is historical,
not current host memory pressure. The standalone monitor must be restored before
any separately authorized experiment launch; do not restart the old supervisor.

The account user slice currently has no enforced CPU/RAM quota. The broker's
150.987 GiB reservation cap and 24-CPU affinity are operational controls; the
proposed 150 GiB systemd hard limit was not applied. Preserve the additional
64 GiB host-memory headroom requirement.

Historical resource audit: 96,192 samples in
`results/parallel_memory_recovery_20260929/resource_history.jsonl` contain no
account RSS sample above 150 GiB. Peak: **115.8415 GiB**, October 2 at 05:10:57 PDT
(line 82130). Sampled RSS is not total cgroup memory accounting and cannot exclude
brief spikes or uncovered periods. Post-reboot kernel peaks cannot prove prior usage.

## Accepted evidence

Registry refresh: **634 accepted records, 70 queues, zero record discrepancies**.
This checks acceptance markers and recorded timings, not a new numerical-oracle run.
FA-50: 22 accepted; FA-51: 54 accepted. FA-51c Qwen: 1 accepted; FA-51c Llama: 4 accepted;
FA-52: 6 accepted. These latest queues are `complete_requires_review`; no paper-readiness
or global tuning optimum is asserted. FA-41 final review verified 24 accepted cases.

Latest recorded full-ROI timings (microseconds); configurations and workloads differ,
so rows are not interchangeable baselines:

| Queue | Case | ROI us |
|---|---|---:|
| FA-51c Qwen | `t4096_spm_bc256_r256_f15_cr16` | 8398.4216 |
| FA-51c Llama | `llama_t4096_spm_bc256_r256_f15_cr16` | 18920.3400 |
| FA-51c Llama | `llama_t4096_cache_bc256_r256_f15_cr16` | 19254.7216 |
| FA-51c Llama | `llama_t4096_spm_bc256_r256_f0_cr16` | 19610.9116 |
| FA-51c Llama | `llama_t4096_cache_bc256_r256_f0_cr16` | 20009.3092 |
| FA-52 Q-only | `head_vonly_t4096_bc256_r512_cr8` | 9652.9392 |
| FA-52 Q-only | `head_qonly_t4096_bc256_r512_cr8` | 10457.1008 |
| FA-52 Q-only | `head_cache_t4096_bc256_r512_cr8` | 10390.4652 |
| FA-52 Q-only | `head_cache_t4096_bc256_r512_cr16` | 8851.0364 |
| FA-52 Q-only | `head_vonly_t4096_bc256_r512_cr16` | 8787.0928 |
| FA-52 Q-only | `head_qonly_t4096_bc256_r512_cr16` | 9458.4268 |

Read the [ledger](LEDGER.md) for inherited kernel, precision, geometry, protocol,
startup and diagnostic limitations. [CURRENT_RESULTS.txt](CURRENT_RESULTS.txt)
contains acceptance-marker paths for every registered case.

## Cleanup and preservation

Audited and recoverably removed 292 generated Python bytecode files whose source
still exists, screening 470 input/manifest files and checking open/mapped account files.
Net allocated storage reclaimed after the recovery archive: **1,667,072 bytes**.
No loose object/static-library/simulator outputs were found in the inspected benchmark,
script, SPM-tool and verification trees outside frozen results. One bytecode file without
a matching source was retained. The 9.5 GiB main gem5 build is the retained incremental
build, not classified as erroneous. Results, isolated build dependencies, executable
binaries, environments, toolchains and all unrelated work remain intact.

302 unique frozen inputs from FA-41, FA-51c and FA-52 matched their manifests before
cleanup and were re-hashed afterward without drift. Recovery files and the exact
removal manifest: [cleanup archive](../../archive/workspace_cleanup_20261007/README.md).
This was a bounded safe cleanup, not a claim that all historical build products are
unneeded or that the pre-existing dirty Git worktree is clean.
