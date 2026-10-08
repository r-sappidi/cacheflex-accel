FA42 scheduling-only parallel continuation

User authorized more concurrency. Preserves running early_cache_on, accepted
small gates and early_cache_off. Six remaining prefixes run independently, each
under an atomic 10GiB reservation and a distinct pair of CPUs from the existing
24/128 allowed set. Trace-on acceptance waits for matched trace-off acceptance.
The exact original scientific validation block is reused verbatim and AST-hashed.
Guest, simulator, configs, stop ticks, trace windows and frozen inputs unchanged.

No cross-namespace process signals. The old serial runner exclusively creates
its next case directory before launch. We atomically claim all six pending case
directories for the parallel workers. When its currently running case completes,
the old coordinator hits that existing directory and exits with FileExistsError
BEFORE launching more work. This specific exception is an expected scheduling
handoff, not a simulator or correctness failure. Other exceptions remain errors.
The original status/evidence is preserved; parallel_v1/status.json is authoritative.
If stopped unexpectedly, preserve claims and inspect manifests before recovery;
never restart the old serial coordinator or another launch.py blindly.

One parent with 2GiB reservation plus six 10GiB workers; initial simulatorAS8GiB
with admitted4GiBgrowth. Shared20%CPU/RAM cap, >=64GiB extra host headroom,
pressure pauses, 2GiBtrace and50GiBdisk guards inherited. Four existing runs
are preserved. Expected peak10simulators and108GiBreserved before growth.

Launch once: python3 benchmarks/qwen/fa/packet_parallel/launch.py
Status: results/cacheflex_flash_attention/packet_diag_20261004/parallel_v1/status.json
Raw cases remain under experiment_v1/runs, each delegated.json records ownership.
Current log FA42 points at the new aggregator; no duplicate acceptance entries.
