# Experiment evidence

- [LEDGER.md](LEDGER.md): authoritative interpretations, validation scope,
  negative results, invalidations and scientific limitations.
- [CURRENT_RESULTS.txt](CURRENT_RESULTS.txt): generated accepted records and
  queue locations. Refresh with `python3 scripts/refresh_current_result_log.py`.
- [current_results.json](current_results.json): machine-readable counterpart;
  `benchmarks/status.py` uses its queue registry and reads status files live.
- [INVENTORY.md](INVENTORY.md) and [inventory.json](inventory.json): searchable
  historical discovery; use `scripts/index_experiment_history.py --query`.
- Other documents in this directory are experiment-specific plans and audits.
  Their launch-time status may be historical; the ledger and live queue files
  take precedence.

Keep scientific outcomes here, implementation details beside the source, and
operating rules in the root `AGENTS.md`. A validator-accepted case is not
necessarily a tuned target or a paper-ready comparison.
