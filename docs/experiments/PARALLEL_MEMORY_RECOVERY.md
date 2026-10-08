# OPS-21: host-memory recovery under the same20% budget

2026-09-29. User authorized increasing the per-process allowance and continuing
all affected work. OPS-20 had imposed4.5GiB RLIMIT_AS per simulator. T16K NoC FA
and T2048 SwiGLU SPM mp8/kc1024 terminated with host bad_alloc/allocation errors;
twelve peers were paused. This is a host-allocation failure, not a simulated
cache-capacity result. AI-12 independently stopped on the upstream failure guard.

Adopt the twelve existing worker/simulator pairs without restarting them; keep
their PIDs, program state, original commands and original acceptance tails.
Stop the old supervisor so it cannot re-pause resumed jobs or publish twice.
Preserve its state and the two failed run directories with file hashes before
restarting only the terminated cases. New worker differs solely in simulator
RLIMIT_AS4.5->8GiB; original command, reference and acceptance ASTs are unchanged.
Original frozen scripts, budget.json, simulators, guests and configs stay intact.

Each admitted simulator receives8GiB AS; each worker reserves1.5GiB. Unadmitted
paused pairs retain4.5+1.5GiB until an upgrade fits. Reserve8GiB outside work,
9.5GiB for existing llama (also increased to8GiB without restart),1GiB recovery
supervisor, and the full potential34GiB of the independent L3-11 queue until its
coordinator exits. Total must never exceed162121272524B (20% host RAM).
All workers retain the same24-of128 CPU affinity; all-user CPU/RSS and available
memory/disk are monitored. Pressure pauses this supervisor's workers; the L3-11
guard reads the published resource state and independently pauses its children.
This is not a privileged per-user cgroup: unrelated future programs cannot be
hard constrained, but outside RSS/CPU is included in the monitoring guard.

With all12 old peers present and L3-11 live, initial reservation is124.5GiB.
Two retries add19GiB; two peer upgrades add7GiB:150.5GiB, below150.987GiB cap.
Other peers remain safely paused until earlier work finishes/releases reservations.
Once L3-11 exits, its34GiB reservation is freed and more peers can resume.

The old journal status/resources paths are maintained by the new single writer
for compatibility with L3-11. Historical failure evidence remains in the recovery
journal and each family's previous_failure field. New failures remain fail-closed
per family; no numerical/configuration check is relaxed or failed timing accepted.

AI-12 continuation is prepared as experiment_memory_v2 and starts automatically
after the recovered targets finish. It reuses59 accepted cases, skips only their
exact case identities, retains all scientific run/input/command functions by AST
identity, and uses8GiB simulator allowances with up to12 admitted children under
the same aggregate accounting. Original experiment_v1 stays preserved, including
its failure record. Remaining phases and selection policy are unchanged.

Root results/parallel_memory_recovery_20260929; authoritative status/resources
there and mirrored in results/parallel_budget_20260929. Failed runs are under
preserved/{long,ffn}; old workers stay under their original journal paths.
No new scientific experiment or performance claim; same L3-10/AI-10/AI-12 work.
