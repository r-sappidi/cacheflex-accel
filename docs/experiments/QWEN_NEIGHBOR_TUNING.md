# L3-12: adaptive T4096 R/Bc neighbor check

2026-09-29. User authorized parallel neighbor checks after L3-11. Related
FA-29 established analytical bounded search; L3-07/08 tested three common Qwen
geometries at smaller sizes; L3-11 fills target-size coverage. Four new L3-11
cases accepted at preparation; last privateR128Bc256 still running. Current
winners allR256Bc256/Br16. This queue imports the final prior result before
adaptive winner selection and never treats an unfinished comparison as final.

First neighbors: R512/Bc256 andR256/Bc128, all three cache/private/L3 backends.
At T4096 the R512 neighbor reduces SPM copies974848->516096 but increases
busiest-core pair work16384000->18087936 and imbalance1.0504->1.0952. Bc128
reduces operand/scratch footprint but doubles L3 commands487424->974848;
it does not reduce query-group causal padding at fixedR256. These are structural
tradeoffs, not predicted latency. Both pass checked guest bounds and existing
15% imbalance/coherent-capacity screens. No identical accepted Qwen cases found.

For every new geometry, all3 full-output T1024 gates precede all3 T4096 cases.
Reuse references when Bc matches; generate an independent Bc128 reference with
the unchanged oracle. Same-Bc exact bits, all-output arithmetic, config/count/
finite-resource/global checks are unchanged (entire run function AST identical).
Same guest_v4/lifecycle build_v5, precision, full ROI, NoC FP16+init L3. No RTL
or direct-path change. The objective is lowest accepted complete FA ROI/backend.

After each batch, select each backend's best measured T4096 configuration.
Take the union of its unmeasured axis neighbors in R{64,128,256,512} and
Bc{128,256,512,1024},Br16 fixed, subject to the same analytical screens.
Measure each selected geometry on all3backends. At most6new geometries total
(18target cases plus18small gates). If the budget is exhausted with a remaining
frontier, explicitly report incomplete neighborhood coverage. Otherwise claim
only axis-local convergence within this screened grid, not global optimality,
not optimization over Br or alternative kernel algorithms. Retain every null
and regressive result and analytical rejection reason.

Resource integration: two new simulators maximum,8GiB AS each,2GiB coordinator,
18GiB shared reservation. Update the recovery supervisor's accounting through
a state-preserving handoff, not by modifying frozen scientific inputs. Its
previous34GiB reservation for L3-11 can shrink to10GiB because only one of its
fixed five tasks remains; no further task can be launched by that frozen runner.
The neighbor reservation persists until completion and coordinator exit. Preserve
all existing simulation PIDs and the same24CPU affinity/20%aggregate RAM ceiling.
Older remainder queue follows after recovery and these FA queues finish.

Root results/cacheflex_l3_spm/qwen_neighbors_20260929. Status, per-iteration
selection, gates and neighbor_decision.json are underexperiment_v1. Only accepted
ROI timings enter comparisons. Admission may wait even when measured RSS is low
because the bound reserves the full permitted virtual allocation of every job.
