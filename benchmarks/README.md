# CacheFlex benchmarks

Start here for workload sources, current experiment entry points, validation, and results. Run commands from the repository root. Run coordinators only after checking live status: many launchers create experiment directories or start simulations immediately.

## Workloads

| Directory | Purpose |
|---|---|
| [llama/](llama/README.md) | Llama-3.2-1B-shaped FA, counter-guided cache/SPM baseline tuning |
| [qwen/](qwen/README.md) | Qwen-shaped FA, SwiGLU FFN, projections, and remaining pipeline components |
| [cacheflex_flash_attention/](cacheflex_flash_attention/) | Shared FA kernels, independent references, tests, analytical models, and historical experiments |
| [cacheflex_flash_attention/llama_cpu/](cacheflex_flash_attention/llama_cpu/) | Llama-shaped attention studies |
| [cacheflex_spm_gemm/](cacheflex_spm_gemm/) | Cache/SPM GEMM and multicast comparisons |
| [acl_sve_gemm/](acl_sve_gemm/) | Shared GEMM kernels and drivers |
| [acl_full_stack_gemm/](acl_full_stack_gemm/) | Full-stack ACL GEMM baseline |
| [ai_workloads/](ai_workloads/) | Shared FFN/MLP kernels, validation, and operator studies |
| [spm_smoke/](spm_smoke/), [spm_store_smoke/](spm_store_smoke/) | Simulator/protocol correctness tests |
| [verification/](verification/README.md) | Current bounded Rumur models and verification results |
| [infrastructure/](infrastructure/README.md) | Shared launchers, resource scheduling, experiment ledger and analysis tools |
| [maintenance/](maintenance/) | Audited repository cleanup utilities |

## Status and evidence

Run `python3 benchmarks/status.py` for a read-only summary of the registered nonterminal experiment queues (`--all` includes completed and historical queues). It never launches or resumes work. It reads each registered status file live, flags records predating the current boot, and checks recorded PID visibility; existing PIDs are not verified process identities. It also reports stale broker heartbeats. Refresh the registry with `python3 scripts/refresh_current_result_log.py`.

Use [the experiment ledger](../docs/experiments/LEDGER.md) for accepted findings, caveats, and superseded attempts. `results/` retains frozen sources, configurations, traces, numerical checks, and accepted results. Benchmark result links point there without moving the evidence.

The `qwen/`, `verification/`, and `infrastructure/` entry points use relative symlinks to preserve existing imports, immutable input hashes, and running coordinators. They are navigation aliases, not copies. Do not edit frozen guest snapshots through these links. New benchmark-specific code should live in its workload directory; shared scheduling and simulator implementation remain separate.

## Cleanup and recovery

The workspace cleanup removed CV sources, PDFs and previews. Obsolete root build products, historical Rumur build products/logs, root build logs, and superseded Rumur launchers are preserved in [the cleanup archive](../archive/workspace_cleanup_20261001/). See its manifests and `complete.json` for the verified scope and recovery command. Current gem5 incremental builds, active/frozen inputs, accepted results and failed diagnostic evidence were retained.

The earlier benchmark-source cleanup is documented in [the previous benchmark map](../archive/workspace_cleanup_20261001/previous_benchmarks_README.md). Recovery archives under `archive/benchmarks_cleanup_20260915/` remain available.

Current FA diagnostics are indexed in [qwen/fa/README.md](qwen/fa/README.md).
Categorized legacy utilities are indexed in [scripts/README.md](../scripts/README.md).
