# Qwen FA cache tuning correction — L3-08

User explicitly requested adding the missing cache tuning immediately. The
earlier cache-freeze restriction is superseded for this new isolated queue.
Do not alter the running L3-07 queue or any of its frozen dependencies.

Related ledger IDs: FA-06/29 established bounded baseline/kernel searches on
older workloads; L3-05 transferred direct PV epilogues to cache as well as SPM.
L3-06 completed Qwen Hq14/Hkv2 causal FA using inherited configurations and one
additional T1024 cache Bc256 control. L3-07 selected two geometries for private
SPM and L3 but omitted cache. Its T1024 R128/Bc256 result improves both SPM arms;
cache has not measured that query grouping. This omission prevents interpreting
the improved SPM-versus-cache ratios as an independently tuned comparison.

Materially new work is cache's Qwen-specific bounded tuning on the exact same
guest_v4/lifecycle build_v5, causal schedule, finite resources and ROI. Existing
accepted cache controls are reused. Scores are uncalibrated sensitivity rankings,
not predicted runtime. Complete ROI latency versus the inherited cache baseline
determines the winner; the final report compares the best accepted result for
each of cache/private-SPM/L3. A faster cache result reduces any claimed SPM gain.

## Candidate selection and budget

scripts/qwen_cache_tuning.py imports the unchanged structural model and applies
the same Br16, R/Bc enumeration, capacity/imbalance screens and144 sensitivity
scenarios as L3-07. Cache footprints include coherent K and V rather than an
SPM V tile. A grouped operand-span proxy represents potential K/V panel reuse;
it is explicitly not actual cache traffic, miss count or DRAM demand.

The highest-ranked noncontrol is R256/Bc256. The second candidate is R128/Bc256,
the diagnostic needed to test the improvement seen in the SPM arms. Including
the inherited R256/Bc512 control, all three backends now cover the SAME three
geometries at T1024/T2048. Each backend has two candidate geometries plus control.
The T1024 cache R256/Bc256 result already exists and is reused, not rerun.

1. New full T1024 gate: cache R128/Bc256. Reuse both R256 cache controls.
2. T2048 cache R256/Bc256 and R128/Bc256. Reuse R256/Bc512 control.
3. Transfer the measured T2048 winner to T4096 only if it improves on the
   inherited baseline. Retain inherited T4096 as an eligible winner.

One new gate + two target-screen cases + at most one transfer. This matches the
private/L3 candidate budget and measured-improvement transfer rule. It is a
bounded best-measured result, not proof of a global optimum or T4096 optimum.

## Isolation, validation and scheduling

New root results/cacheflex_l3_spm/qwen_cache_analytical_20260928. The runner is
derived from the frozen L3-07 runner. AST audits require run(), work() and
get_reference() to be identical; phase() differs only by max_workers2->1.
Full independent oracle, dense small-case check, exact same-Bc outputs, config,
copy/command counts and finite occupancy/global-guard acceptance are retained.
The completed private/L3 results and original controls are frozen/reused when
the new queue starts, allowing its report to show all three best-measured arms.

The coordinator is queued immediately. Six existing simulations currently use
the allowed slots. Cache simulations wait for L3-07 successful completion, then
run one at a time alongside at most two FFN simulations and the three older
jobs: at most six in total. Existing CPU-idle/RAM/disk launch and runtime guards
remain active. No simulation is cancelled, restarted or reconfigured. Failure
of the predecessor blocks this queue; failure of any new acceptance gate stops
its downstream cases. No automatic tolerance relaxation or retry.

Status: coordinator.json and experiment_v1/status.json under the new root.
Final comparison: experiment_v1/REPORT.md and effects.json. Earlier L3-07
reports retain their historical frozen-cache scope; this new report supersedes
that scope for the three-way comparison once cache tuning is accepted.
