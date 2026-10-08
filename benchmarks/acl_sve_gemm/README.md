# Arm Compute Library SVE GEMM baseline

This benchmark vendors the generic FP32 SVE interleaved microkernel
`sve_interleaved_fp32_mla_8x3VL` from Arm Compute Library commit
`a281025b89eb08ee733bedf370593d3a9f13f92d`:

`src/core/NEON/kernels/arm_gemm/kernels/sve_interleaved_fp32_mla_8x3VL/generic.cpp`

Upstream: <https://github.com/ARM-software/ComputeLibrary>

The vendored source retains Arm's MIT license header. `acl_kernel.cpp` keeps
the upstream instructions unchanged and adds local explanatory comments. The
local driver packs A as K groups of eight FP32 values and B as K groups of
`3 * SVE_VL` FP32 values, matching ACL's interleaved kernel contract. It reads
the runtime vector length with `svcntw()`, so the output tile is 8x12 at 128
bits and 8x48 at 512 bits.

Build and run the eight-core Neoverse V2/Garnet baseline:

```bash
source setup_spm_env.sh
bash benchmarks/acl_sve_gemm/build.sh
bash benchmarks/acl_sve_gemm/run_mesi3_garnet.sh 64 96 128 1
```

Arguments are `M`, `N`, `K`, and measured kernel iterations. Input loading,
packing, and scalar reference calculation are outside the measured ROI. The
runner uses gem5's public-information `NeoverseV2` timing model, Ruby's stock
`MESI_Three_Level` protocol and an eight-core 2x4 Garnet mesh. Native 128-bit
SVE is the default; set `SVE_VL_BITS` to select another architectural width.
The Neoverse V2 fetch-directed/decoupled front end is enabled by default, as it
is in gem5's full public-information Neoverse V2 configuration.
Widths such as 512 bits are experimental overrides and should be described as
Neoverse-V2-derived wide-SVE models rather than exact hardware models.
The repository's Neoverse V2 functional-unit descriptors are extended with
gem5's SVE predicate and vector-memory operation classes. This supplies the
missing issue paths for `ptrue`, `ld1w`, `ld1rqw`, and `st1w` while preserving
the model's functional-unit counts and two-cycle load/store latency.

Run all seven model-layer shapes with:

```bash
bash benchmarks/acl_sve_gemm/run_sweep.sh
```

Results are written under
`results/acl_sve_gemm/mesi3_neoversev2_fdp1_vl128_garnet_8c_*`. Existing generic
`DerivO3CPU` results remain in their original `mesi3_o3_garnet_8c_*`
directories, so the two models cannot overwrite one another.

The runner enables gem5 CPU progress reports at `50kHz` and captures all
simulator output in `run.log`. Monitor a run from another terminal with:

```bash
tail -f results/acl_sve_gemm/mesi3_neoversev2_fdp1_vl128_garnet_8c_m64_n96_k128_i1/run.log
```

The progress frequency is configurable. For example, use fewer updates during
a long sweep with:

```bash
PROGRESS_INTERVAL=10kHz bash benchmarks/acl_sve_gemm/run_sweep.sh
```

Set `PROGRESS_INTERVAL=0Hz` to disable periodic CPU progress messages. Epoch
boundaries continue to print as `GEMM_EPOCH` records regardless of this value.
Set `ENABLE_FDP=0` to run the conventional-fetch baseline. FDP-enabled and
disabled results are stored under distinct `fdp1` and `fdp0` prefixes.

Run the explicit 512-bit research configuration with, for example:

```bash
SVE_VL_BITS=512 bash benchmarks/acl_sve_gemm/run_sweep.sh
```
