FA-39 parallel T1024 screening (user authorized 2026-10-04).

T4096 queue remains unchanged. Same guest_v1/simulator, Br16/Bc256/R256,
Hq14/Hkv2/d64, eight cores, causal, FP16/FP32 semantics. Twelve screening cases:
cache/private pairs for control, stage, L1D16, privateL240, both, packed producer.
Each case waits for and records the corresponding accepted T256 gate from the
existing experiment_v1. No duplicated correctness gates, exact controls, or rebuild.
T1024 interpretation remains separate from T4096 capacity/ownership effects.

Runner derived from FA39; command, config audit, pressure, parameter, work and argv
functions have identical ASTs (derivation.json). Acceptance/oracle/bitwise/work/
copy/bounds checks unchanged. Failure in an unmet prerequisite stops this queue.
New coordinator/sim uses two other CPUs within shared24/128 affinity,10GiB initial
lease, same controlled growth and20%CPU/RAM plus64GiBspare enforcement. One new sim.

Status/report: results/cacheflex_flash_attention/bottleneck_20261004/screening_t1024_v1.
Launch: python3 benchmarks/qwen/fa/bottleneck_1024/run.py
