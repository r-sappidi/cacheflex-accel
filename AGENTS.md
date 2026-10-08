# CacheFlex multicore: agent guide

Updated 2026-10-07. This is the operational guide, not a chronological run log.
User instructions take precedence. Preserve unrelated work in this dirty worktree.

## Start here

- `README.md`: repository map.
- `benchmarks/README.md`: workload and infrastructure entry points.
- `benchmarks/qwen/fa/README.md`: FA experiments and diagnostic lineage.
- `docs/experiments/LEDGER.md`: authoritative accepted findings and limitations.
- `docs/experiments/CURRENT_RESULTS.txt`: generated validator-accepted records.
- `docs/experiments/CURRENT_STATE.md`: dated post-reboot audit and cleanup scope.
- `scripts/README.md`: categorized legacy tools; names are not launch authorization.

Run from the repository root:

```bash
python3 scripts/refresh_current_result_log.py
python3 benchmarks/status.py
python3 scripts/index_experiment_history.py --query 'your topic'
```

## Required experiment workflow

1. Read relevant ledger entries, search the inventory, refresh the current result
   log, and inspect live queue status before proposing or launching work.
2. Cite prior ledger IDs. State what changes, why prior evidence is insufficient,
   and which full-ROI outcome would change the decision. Avoid duplicate runs.
3. Prefer analytical capacity/work/traffic screening followed by counter-guided
   iteration and bounded parameter neighbors. Do not default to blind sweeps.
4. Reuse valid results. An SPM-only protocol change does not automatically require
   retuning coherent cache; check compatibility and replay narrowly if necessary.
5. Preserve full numerical oracle, output, configuration, work/copy, resource and
   frozen-input checks. Do not loosen acceptance criteria to finish a run.
6. Record positive, negative, failed and cancelled outcomes with caveats in the
   ledger. Register new queues in `scripts/refresh_current_result_log.py`.

Acceptance is not paper readiness. Distinguish correctness gates, diagnostic
idealizations, target timings and tuning winners. Never silently mix guests,
hardware profiles, precision, shapes, protocol versions or startup conditions.
A bounded search is not proof of a global optimum. Component projections are not
an executed end-to-end model. Do not add per-core times as wall time, interpret
pending requests as whole-core stalls, or treat phase duration as removable cost.

## Resources and process safety

- All experiments together must remain below **20% of server CPU and RAM**,
  with **64 GiB additional host-memory headroom**. Account for all user processes.
- Existing shared CPU affinity is 24 of 128 host CPUs; RAM reservation cap is
  approximately 150.987 GiB. Recheck the host and broker, not just these numbers.
  These are operational budgets, not an account-wide kernel quota; the proposed
  150 GiB hard cap had not been applied at the October 7 audit.
- Shared reservations: `results/parallel_overlap_20260930/broker.json` and its
  `broker.py`. Use admitted memory growth, quota guards and the existing broker;
  independent per-run caps are insufficient.
- Resource monitor: `results/retained_protocol_restart_20261002/spm_only_resources.json`.
  Written by `benchmarks/infrastructure/resource_monitor.py` (OPS-25); after a host
  reboot, restart it and clear leases whose owner PIDs are gone, or nothing admits.
- Do not signal a PID from an old note. Check host command line, start time and
  ownership; namespace PIDs can differ. Status-file labels alone do not prove life.
- Never resume old stopped supervisors or cancelled Qwen long-context queues.
  No restart, duplicate coordinator or new experiment is implied by a status check.

## Frozen evidence and code organization

- `gem5/`: simulator implementation; retained builds may use isolated source trees.
- `benchmarks/qwen/`, `benchmarks/llama/`: new workload-specific development.
- `benchmarks/infrastructure/`: shared scheduling and admission helpers.
- `scripts/`: legacy paths required by imports, frozen manifests and replay tools.
- `results/`: immutable experiment inputs, guests, model snapshots and evidence;
  live status files belong to their coordinators. Never reorganize these in place.
- `verification/rumur/current/`: bounded protocol models.
- `archive/`: recoverable superseded artifacts and cleanup manifests.

Do not edit frozen dependencies, replace protocol files from another checkout,
reset the worktree, remove result evidence, or rebuild over binaries used by jobs.
Use isolated source/build/guest versions for scientific changes. Factor reusable
code for new experiments only after checking dependencies; preserve historical
runners and exact evaluation behavior. Compatibility symlinks are intentional.
Check manifests before and after cleanup. Do not indiscriminately delete builds,
virtual environments or caches that running jobs may need.

## Protocol and evaluation context

Retained-source SPMCP copies without releasing coherent source ownership:
a valid coherent source retains its state/registration; an absent source uses
GETS_SILENT without adding a sharer. SPMWB still requires coherent invalidation and
ordering. Do not restore old source-release behavior or assume all bridges are
coherence-neutral. Lazy migration handles physical cache/SPM slot conversion.
See VER-02 and the retained-protocol source snapshot for precise semantics.
Rumur v6 passed all 15 bounded positive/negative cases; this is not a full-machine
proof or a replacement for runtime correctness checks.

Qwen FA is causal GQA with 14 query heads, 2 KV heads and head dimension 64;
Llama-3.2-1B-shaped FA uses 32 query heads and 8 KV heads. Both are component
benchmarks. Kernel geometry and precision must be taken from each accepted case.

## Current evidence pointers (refresh before reporting)

- **FA-35:** retained-protocol 4K SPM matrix completed and validated:
  `results/retained_protocol_restart_20261002/fa_matrix_spm_only/status.json`.
- **FA-41:** Llama adaptive tuning continuation:
  `results/ordered_fa_followup_20261005/llama/status.json`.
  Final review completed October 6 (24 accepted); see `llama/FINAL_REVIEW.md`.
  Its historical status label still says `awaiting_final_review`.
- **FA-34/36/38:** optimistic ring scheduling studies; zero-time transfer is
  schedule-specific headroom, not a finite-link hardware result.
  FA-34/38 bulk files archived (OPS-24, archive/ring_mechanism_20261005); FA-36
  and FA-37 ring dependencies remain in place.
- **FA-39–45:** transport, staging, packet-latency and software-control diagnostics.
  Read the ledger to avoid repeating these or mixing failed and accepted attempts.
- **FA-46/47/48:** blocking then asynchronous PV diagnostics. FA-48's inert
  controls remove a startup confound; its clean finite-service/fewer-fence arm
  reduces T1024 latency 3.929% versus matched ordinary SPM. Functional V access
  and absent bank/NoC arbitration prevent a realistic hardware speedup claim.
- **FA-49:** all 8 ordinary cache/SPM OFF/ON phase-profiler cases validated:
  `results/cacheflex_flash_attention/phase_profile_20261005/`.
  T1024 matched diagnostic guest: cache 1000.4952 us, SPM 950.7684 us.
  Critical-core SPM softmax 316.0212 us, PV+accumulation 213.1364 us,
  QK 198.2004 us, staging/setup 28.9284 us. Not a newly tuned baseline.
  Per-phase counters cover copies, physical loads, RAW waits and port rejects;
  cache-level misses, NoC and SIMD counters are not all phase-attributed.
  Read `REPORT.txt`, `phase_counters.csv` and `FINAL_VERIFICATION.json`.

- **FA-50/51:** software optimization and bounded Bc/R tuning finished (22/54 accepted).
- **FA-51c/52:** completed Qwen/Llama 16-credit follow-up (1/4 accepted) and Q-only
  placement study (6 accepted); `complete_requires_review`. See current result log.
- **OPS-26:** October 7 post-reboot audit found no live experiments or monitor.
  AI-13/13b/FA-51b active entries and broker heartbeat are historical; the status
  command flags pre-boot records. No cleanup action authorizes a queue restart.

- **FA-53:** exact latest Qwen SPM winner phase replay:
  `results/cacheflex_flash_attention/winner_profile_20261007/status.json`.
  Complete and validated: full ROI 8398.4216 us, exact FA-51c output/non-host stats.
  Critical-core softmax 38.49%, PV 24.72%, QK 19.86%, P packing 5.65%, staging
  3.88%, output accumulation/rescaling 2.58%. Read `REPORT.md` and
  `FINAL_VERIFICATION.json`; elapsed phase intervals are not removable costs.

## Maintaining these instructions

Keep this guide concise. Put new outcomes in the ledger, live state in queue
status files, and implementation details beside their source. Update this guide
only when operational rules, navigation or current evidence pointers change.
The previous 368 KB chronological guide is preserved verbatim at
`archive/workspace_cleanup_20261005/before/AGENTS.md`; it is historical context,
not authorization to run its old pending tasks. Lowercase `agents.md` points here.
