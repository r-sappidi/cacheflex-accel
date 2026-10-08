# Historical experiments and integration checks

These scripts retain earlier investigations, not the current recommended
best-to-best launch path. They often depend on frozen, dated result directories
and some execute immediately at module scope. **Do not import them for discovery
or launch them collectively.** Root-level shared builders/analyzers remain in
use; `_paths.py` makes them available when running these scripts directly.

| Scripts | Original question |
| --- | --- |
| `run_correct_fa.py`, `run_correct_pinned.py`, `run_correct_residual.py` | Corrected numerical controls and pinned/driver residual comparisons |
| `run_correct_rich_final.py`, `run_correct_software.py` | Rich-input validation and corrected software comparisons |
| `run_correct_trace.py` | Trace the corrected comparisons |
| `run_driver_residual.py`, `run_aligned_driver_causes.py`, `run_residual.py` | Driver organization and residual slowdown ablations |
| `run_pv_alignment.py`, `run_qk_alignment.py` | Code-alignment sensitivity |
| `test_fa_export_run.py` | Real gem5 controls for the post-ROI export fix; launches simulations |
| `test_fast_trace_events.py TRACE [N]` | Compare old/new trace readers using an existing trace |

Run from any working directory using the script's full path. Positional
arguments and dated dependencies remain as specified in each script. All moved
files previously lived one directory above this one; historical logs were not
rewritten. Moving them does not change the validity of their old measurements.

For current work see [the FA index](../README.md). For historical study context,
see [causal ablations](../docs/CAUSAL_ABLATIONS.md),
[driver causes](../docs/DRIVER_CAUSES_PLAN.md), and
[residual study](../docs/RESIDUAL_STUDY.md).
