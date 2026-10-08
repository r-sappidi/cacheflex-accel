# L3-11: complete the Qwen T4096 candidate comparison

2026-09-29. User authorized the targeted target-size tuning discussed after
the baseline audit. Related L3-06/07/08 already validated and searched the same
FP16 causal Qwen-shaped FA at T1024/T2048. Their T4096 evaluation retained
inherited configurations and transferred smaller-size winners. In particular,
cache R128/Bc256 took11015.6924us, slower than inherited R256/Bc51210527.9640us.
This is evidence that transferring the winning configuration does not establish
a target-size optimum. L3-09 direct-link results are a distinct transport arm.

## Frozen workload and equal candidate set

Batch1, Hq14/Hkv2, d64, T4096, eight cores, Br16, same guest_v4 and lifecycle
build_v5, finite profile, precision/rounding, scheduling and full ROI. Every
backend gets the same three geometries: R128/Bc256, R256/Bc256, R256/Bc512.
Reuse the four accepted T4096 measurements and all prior validation evidence.
Run exactly five missing cases:

| Backend | R | Bc |
|---|---:|---:|
| cache |256|256|
| private-SPM |128|256|
| private-SPM |256|512|
| L3 reduction |128|256|
| L3 reduction |256|512|

L3 is the existing NoC FP16-contribution/first-initialization arm from L3-07/08,
not a new hardware mechanism or an unmeasured direct-link result. Do not mix
transport changes into this tuning comparison. No new guest or simulator build.
The two required T4096/Bc256 and Bc512 full independent references already exist
and are hash-verified/reused. Same-Bc outputs must match existing accepted bits.

## Acceptance and interpretation

Derive the runner from the completed L3-08 runner. Audit AST identity for the
entire post-simulation acceptance suffix, workload counts, reference generation,
and command construction. Changes are output paths, fixed case selection,
report labels, and host scheduling/resource controls only. Preserve independent
full-output checks, exact peer bits, frozen-file checks, all-field configuration
audit, command/copy counts, finite LSQ/source/sink/bank bounds, and global guard.
Any new simulation or validation failure stops this queue and preserves partials.

Select each backend's lowest accepted complete FA ROI, including old controls.
Report all positive/null/regressive outcomes. Faster cache/private results become
the denominators for L3 benefit. Completion establishes an equal three-candidate
target-size comparison, not global optimality. Review target-size structural
capacity/imbalance and measured trends before any additional neighbor search;
no automatic broad sweep or direct-path transfer is part of this five-case queue.

## Host resources and isolation

Use the existing 24-of128 CPU affinity (18.75% host CPU ceiling for the experiment
group), 20% host-RAM budget, and all-user CPU/RSS monitoring. Maximum four new
simulators, 8GiB address-space limit each, plus2GiB for the coordinator. Reserve
the old scheduler's maximum future outstanding jobs at6GiB each, including
paused/failed-family jobs, plus6GiB for llama and8GiB outside-work margin. Admission
rechecks fresh old status and live resource data; pressure pauses only this new
queue's children. It never resumes or modifies old workers/coordinators.

Preflight found host-allocation failures in separate long-context/FFN families
under their4.5GiB process limit; their peers are paused by the old scheduler.
AI-12 stopped after59 accepted gates on the old-queue failure guard. These are
separate failed/partial experiments, not failed T4096 FA controls. Preserve them.
The new reservation is larger per simulator and still fits the aggregate cap.

Root: results/cacheflex_l3_spm/qwen_t4096_tuning_20260929. Authoritative status:
experiment_v1/status.json; host guard: experiment_v1/resource_status.json.
