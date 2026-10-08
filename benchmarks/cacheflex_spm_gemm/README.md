# CacheFlex SPM SVE GEMM

This is the SPM-optimized counterpart to `benchmarks/acl_sve_gemm`. It keeps
the same 8x3VL FP16 tile, packed A/B/C formats, eight-thread work assignment,
ROI boundaries, functional input initialization, and validation.

The kernel is specialized for 512-bit SVE (96 FP16 columns per tile). For each K block, every core copies
the shared packed B panel into its private SPM once, placing the three 64-byte
B vectors in ways 0, 1, and 2. The core then reuses that panel across all M
tiles assigned to it. K dimensions larger than 255 are processed in blocks;
later blocks load and accumulate the partial C tile.

The address-aware O3 shadow table orders speculative SPM loads against only
overlapping older copies, including the wrapped set-256 address.

Build and run:

```bash
source setup_spm_env.sh
bash benchmarks/cacheflex_spm_gemm/build.sh
bash benchmarks/cacheflex_spm_gemm/run_mesi3_spm_garnet.sh 64 192 128 1
```

For the most controlled comparison, run the ordinary ACL kernel through the
same SPM-capable gem5 binary and protocol with zero SPM ways:

```bash
bash benchmarks/cacheflex_spm_gemm/run_matched_acl_garnet.sh 64 192 128 1
```

The fair baseline is the existing ACL runner with `SVE_VL_BITS=512`; this
runner matches all of its CPU, cache, Garnet, clock, threading, input, and ROI
settings. The SPM run reserves three of the 16 private-cache ways. Copy-in,
barriers, K-block accumulation, and reduced coherent-cache capacity are
deliberately included in its measured ROI.

The runner models two SPM load issue ports and one SPM store issue port per
core, matching the Neoverse V2 model's two load-capable and one store-capable
LSU pipelines.
