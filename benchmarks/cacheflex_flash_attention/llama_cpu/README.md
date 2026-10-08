# Pinned llama.cpp CPU FlashAttention baseline

Upstream: https://github.com/ggml-org/llama.cpp, commit
`4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0`. GCC 15.2, static AArch64
armv8.2-a+sve+fp16 build; no BLAS, KleidiAI, OpenMP or GPU backend.
ggml's own pthread pool, task scheduler and tiled attention remain in use.

Two explicitly separate implementations:

* `upstream_sve_ready`: unmodified upstream ggml, compiled with SVE enabled.
  At this revision `simd-gemm.h` has no SVE branch and selects the scalar
  source fallback for the tiled QK/PV GEMMs. The compiler may vectorize it.
* `ported_sve_ready`: the same library with one added SVE FP32 GEMM helper
  branch (`sve_gemm.inc`). Eight row accumulators, runtime vector length,
  predicated tails; no changes to attention scheduling, softmax, packing,
  tile sizes or thread count. This is an adapted library baseline, NOT an
  untouched upstream SVE kernel. All old source and binaries are retained.

The external interface is FP16 Q/K/V and FP16 output, with exactly the same
deterministic input values as the persistent-V experiment. ggml needs FP32 Q
and returns FP32 output; both conversions are measured in ROI. Its GEMMs and
online state accumulate in FP32, unlike the custom FA's FP16 GEMMs. We do NOT
claim identical precision, instruction work or storage footprint. These
differences must accompany any external-library comparison.

First invocation after initialization, not guaranteed cold. Input construction,
arena/scratch allocation, output export and validation are outside ROI. A
disjoint add graph completes threadpool startup before measurement, touching
neither FA inputs nor FA kernels. All FA packing, computation, conversions,
dispatch and completion synchronization are inside ROI. Default upstream
threadpool polling is retained. The initial `experiment` used a harness without
this readiness step; it is superseded by `experiment_v2`, with all evidence kept.

The exact existing KV experiment's 8-core VL16 hardware is reused:
64 KiB L1I/D, 512 KiB private L2/core, 4 MiB LLC (8 banks), fixed 8-router mesh,
128-bit links, 8 DRAM controllers, frozen MICRO resource/timing adapter,
nonbinding 1024 sequencer guard, retention on. No simulator changes. Every
serialized config field is compared against the accepted custom cache run,
allowing only executable and argv changes.

Each implementation has an independent gated chain: T128, T512, T4096, T16384;
at most two simulations. Every FP16 AND FP32 output is checked against a
chunked dense float64 reference. Existing 2e-6 + 1% per-element tolerance and
<1% relative L2 apply; output/printed hashes and zero SPM requests checked.
Disassembly is cross-referenced with retired-PC counters to verify actual SVE
execution, not merely binary inclusion. A report must distinguish GEMM SVE
retirement from conversions/softmax SVE retirement.

Artifacts: `results/cacheflex_flash_attention/llama_cpu_20260925/`.
Build with `build.py`, then `ready_build.py`; launch `run.py` using the NumPy
environment and OPENBLAS_NUM_THREADS=1. Status: `experiment_v2/status.json`.
Failed standalone CMake attempt retained; wrapper CMake uses ggml as a
subdirectory, its normal supported integration, without editing its build files.
