# CacheFlex multicore

Start with [benchmarks/README.md](benchmarks/README.md) for workload organization, sources, current runners, validation and result locations.

- [Qwen-shaped components](benchmarks/qwen/README.md): FA, SwiGLU FFN, QKV/output projection and remaining pipeline operators.
- [Experiment ledger](docs/experiments/LEDGER.md): accepted results, failed/superseded cases and evaluation constraints.
- [Verification](benchmarks/verification/README.md): bounded Rumur models and current verification queue.
- `gem5/`: simulator and protocol implementation; `gem5/build/` is the retained incremental build tree.
- `results/`: frozen experiment evidence, including source/configuration snapshots and traces.
- `archive/`: recoverable retired artifacts and cleanup manifests.
- [AGENTS.md](AGENTS.md): concise operating rules and current evidence pointers.

Latest operational snapshot: [October 7 state and cleanup](docs/experiments/CURRENT_STATE.md).

Benchmark navigation uses compatibility links where existing paths are required by active or reproducible experiments. Do not rename frozen inputs or launch duplicate coordinators.

Read-only active queue view: `python3 benchmarks/status.py`; add `--all` for history.
Refresh accepted records first with `python3 scripts/refresh_current_result_log.py`.

Legacy utilities: [scripts/README.md](scripts/README.md). Historical agent notes
and pre-cleanup files: [2026-10-05 cleanup](archive/workspace_cleanup_20261005/README.md).
