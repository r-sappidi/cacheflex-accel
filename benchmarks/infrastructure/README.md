# Shared experiment infrastructure

[scripts/](scripts/) provides the existing shared runners, resource schedulers, source checks, analyzers, and historical implementation utilities. [experiment_docs/](experiment_docs/) provides the curated ledger, plans, and evidence inventory.

These are compatibility links. Physical relocation of the shared scripts is deferred because active coordinators, frozen manifests and historical imports depend on those paths. Current workload entry points are indexed under `benchmarks/qwen/`; not every older `run_*.py` is a current or safe launch command.

Keep the 20% all-user CPU/RAM cap and shared memory-reservation broker when starting experiments. A stopped historical coordinator is not authorization to resume it. Current status files and the ledger take precedence over old PID notes.
