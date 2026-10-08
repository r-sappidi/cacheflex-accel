# Verification

[rumur/current/](rumur/current/) contains the current private-SPM coherence model and focused network-progress/L3-ordering models. [rumur_results/verification_v5/status.json](rumur_results/verification_v5/status.json) is authoritative for the current rerun. [rumur_runner.py](rumur_runner.py) uses shared resource admission; do not duplicate it.

These are bounded abstractions, not a complete eight-core implementation or RTL proof. Current scope and negative controls are documented in the model README and ledger VER-01. Legacy models remain under `rumur/`; their old generated build artifacts were archived with hashes in `archive/workspace_cleanup_20261001/retired.tar.gz`.
