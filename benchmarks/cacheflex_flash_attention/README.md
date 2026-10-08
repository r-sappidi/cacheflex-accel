# FlashAttention benchmarks

This directory contains the custom multicore FA operator, cache/SPM variants,
correctness checks, tuning tools and diagnostics. Start with
[the implementation guide](docs/IMPLEMENTATION.md) to understand the dataflow.
These are operator experiments, not a complete transformer or production
attention library.

## Directory layout

```text
cacheflex_flash_attention/
  README.md             Start here
  docs/                 Implementation guide and diagnostic study notes
  tests/                Host unit tests and standalone SVE test sources
  experiments/          Historical diagnostic runners and integration checks
  *.cpp, *.hpp, *.S     Shared guest sources (stable build paths)
  build_*, make_*       Guest builders and kernel generators
  run_*, sweep_*,       Current runners and reusable experiment infrastructure
  optuna_*, queue_*
  analyze_*, validate_* Counter/trace analysis and validation
```

The root retains shared modules and build inputs because current builders and
running queues resolve them by path. Historical filenames alone do not imply
that a module is unused: `queue_correct_best.py` supplies configuration helpers
to Optuna, and `run_causal.py` supplies hardware audits to current runners.

## Which files should I read or use?

For the new true single-head cache/SPM experiment and live run status, see
[the single-head guide](docs/SINGLE_HEAD.md) and `run_single_head.py`.

The V-staging sweep has been cancelled in favor of
[persistent per-core Q staging](docs/Q_RESIDENT.md), using `run_q_resident.py`.

| Task | Entry points |
| --- | --- |
| Understand the current shared cache/SPM driver | `spm_multicore.cpp`, [implementation guide](docs/IMPLEMENTATION.md) |
| Build corrected native VL16 variants | `build_correct_search.py`, `make_search_kernels.py` |
| Build full-width PV variants | `build_pv_improvements.py`, `make_pv_fullwidth.py` |
| Run current long-context comparisons | `run_large_t_pv.py` |
| Run the separate LLC-retention hardware experiment | `run_llc_retain.py` |
| Tune software configurations | `optuna_fa.py`, `optuna_fa_export_safe.py`, `queue_correct_best.py` |
| Independently check numerical results | `fa_independent_reference.py` |
| Safely export post-ROI results | `fa_export_patch.py`, `fa_export_hook.S` |
| Analyze traffic and attributed core time | `analyze_traffic.py`, `analyze_attribution.py`, `validate_attribution.py` |
| Analyze buffer residency/evictions | `analyze_residency.py`, `analyze_spm_residency.py`, [profiling guide](docs/PROFILING.md) |
| Estimate work and staging counts | `fa_analytical_model.py` |
| Revisit older causal experiments | [experiments/README.md](experiments/README.md) |

Several runners embed dated result dependencies and create new output
directories. Read their arguments/defaults before launching; do not run all
`run_*.py` files as a test suite. Existing long-context queues should be checked
before starting duplicate simulations. Accepted timings and numerical gates
live under `../../results/cacheflex_flash_attention/`, not this directory.

## Testing

From the repository root:

```sh
.venv-fa-optuna/bin/python -B benchmarks/cacheflex_flash_attention/run_tests.py
```

The environment supplies NumPy and Optuna. Use `--pattern test_residency.py`
to select a subset. This runner discovers only host unit tests; the historical
gem5 export experiment and trace-file comparison are deliberately excluded.
The two C++ tests in `tests/` require an ARM SVE toolchain/runtime and are not
host Python tests. The conversion diagnostic demonstrates a conversion idiom;
it is not the independent full-operator numerical acceptance check.

## Refactor scope

The September 15 organization moves study documents into `docs/`, unit tests
into `tests/`, and eleven historical diagnostic runners plus two integration
checks into `experiments/`. Their basenames are unchanged. Invoke moved scripts
using the new paths; their shared-source and repository paths still resolve to
the original locations. Older result logs retain their original command paths.

No guest/kernel code, numerical behavior, timing configuration, source-hashed
build inputs, or active queue entry points were changed. Generated binaries
were archived during the preceding cleanup; older runners may require a rebuild
or restoration as described in [the benchmark index](../README.md).
