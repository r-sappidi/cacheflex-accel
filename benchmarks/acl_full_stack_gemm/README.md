# Full-stack ACL W7 baseline

This harness calls upstream Arm Compute Library `NEGEMM` for FP16 W7 and lets
ACL perform kernel dispatch, packing, multicore partitioning, and scheduling.
Only tensor setup, the thread-count request, and gem5 ROI markers live in the
local harness.

The validated build used ComputeLibrary commit
`a281025b89eb08ee733bedf370593d3a9f13f92d`:

```bash
git clone https://github.com/ARM-software/ComputeLibrary.git /path/to/ComputeLibrary
git -C /path/to/ComputeLibrary checkout a281025b89eb08ee733bedf370593d3a9f13f92d

prefix="$PWD/tools/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-"
cd /path/to/ComputeLibrary
scons -j8 Werror=0 debug=0 asserts=0 neon=1 opencl=0 \
  arch=arm64-v8.2-a-sve os=linux build=cross_compile \
  toolchain_prefix="$prefix" cppthreads=1 openmp=0 examples=1 \
  gemm_tuner=0 standalone=1 data_type_support=fp16,fp32

cd /data2/rsappidi/cacheflex_mc
ACL_ROOT=/path/to/ComputeLibrary bash benchmarks/acl_full_stack_gemm/build.sh
bash benchmarks/acl_full_stack_gemm/run_w7_vl16_8core_ruby.sh
```

`GEMMInfo::fast_math` is enabled because ACL gates its SVE1 FP16-accumulate
kernels on that public option. The guest confirms `sve=1`, `sve2=0`, and
`fp16=1` before ROI.
