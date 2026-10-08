# FA-50: softmax/PV software optimizations

One ELF derived from the accepted FA-39 guest (matrix guest_v3 lineage), runtime flag
word argv[17]; flag 0 runs the verbatim FA-39 softmax path. Retained simulator and
configuration unchanged. Flags: 1 four-row interleaved softmax; 2 PV epilogue without
alpha multiply when all alpha==1 or first KV block; 4 row max from an fmaxv QK
epilogue (non-boundary blocks), skipping pass 1; 8 causal mask folded into pass 1;
16 Q pre-scaled by 0.125 (exact barring FP16 subnormals; tested, not assumed).
Flags 0/1/2/4/8/15 must reproduce FA-39 output bits exactly; 16/31 need the full
oracle and record bit equality. 16 T256 gates (both backends), then T1024 cache/SPM for
flags 0, 15 and 31, each only after both of its gates pass. Four lanes (budget CPU
offsets 12-19), broker leases, pressure pause, 900 s progress watchdog.
Results: results/cacheflex_flash_attention/softmax_opt_20261005/REPORT.md.
