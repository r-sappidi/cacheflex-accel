# FA-53: exact latest-winner phase replay

Authorized October 7 after the LUT discussion. One Qwen SPM T4096 replay of FA-51c:
Br16/Bc256/R256, flag15, 16 credits, 14Q/2KV, d64, eight cores. Reuse the exact
accepted guest and FA-49 passive observer binary; no guest or simulator rebuild.
No cache/Llama retuning or old queue restart. FA-49's T1024 percentages do not cover
this winner; FA-51c has no saved phase timing. Outcome: identify the dominant
full-ROI phases before evaluating a softmax/packing ingress operation.

`prepare.py` derives an audited map for the frozen executable (symbol/address and
instruction checks). Existing labels identify main phases; disassembly boundaries
separate P packing, PV dot product, and output conversion/rescaling/accumulation
inside existing fused PV functions. Q packing includes task initialization; QK
includes flag15 row maxima. No added guest instructions or global stats resets.

`run.py` uses the shared broker/monitor, 10 GiB initial reservation, 8 GiB initial
address-space cap with admitted growth, two CPUs in the existing shared affinity,
64 GiB additional host headroom, 900-second instruction-progress watchdog and
six-hour wall bound. Accepted FA-51c is the observer-OFF control: require exact
all-nonhost statistics and output bits, configuration, full numerical oracle,
work/copy/credit/LSQ/resource and frozen inputs. Phase timelines must partition
workbegin/workend and entry counts match tasks/row updates; request lifetimes
must conserve and reconcile with existing hardware counters. Never loosen gates.

`analyze.py` reports the last-arriving core's percentages against outer ROI plus
per-core data, overlap counters, chart and verification receipt. Elapsed phase
intervals include stalls and are not removable costs. The epilogue split describes
retirement intervals inside a fused function; out-of-order overlap remains.

Results: `results/cacheflex_flash_attention/winner_profile_20261007/`.
Do not rerun an existing coordinator. Inspect its status and live PID identity.
