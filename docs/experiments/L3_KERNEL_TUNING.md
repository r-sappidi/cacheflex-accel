# L3 kernel tuning, 2026-09-28

User authorized sequential implementation after checking overlap and server use.
Related ledger: FA-06/07/29 (kernel and tile tuning), FA-25–27/32 (output
residency/offload), L3-02/03/04 (bounded reduction, FP16 transport, lifecycle).
The prior L3 experiments retained the PV store/reload temporary and inherited
query tiling. They did not implement register-to-L3 reduction submission.
Earlier tile/group searches are prior art; this is their transfer to the L3
cost model, not a new tiling idea. Bank pipelining alone was approximately null;
FP16 transport and initialization helped; write-through finalization regressed.

Source and inventory searches for epilogue/direct-register reduction found no
prior implementation. The checked lifecycle driver stores 16 FP16 PV rows to
`pv`, then reloads each row to submit its reduction. This is distinct from
direct softmax-to-packed-probability fusion, which is outside this experiment.

1. Replace only the PV epilogue with direct commands from live FP16 registers.
   Preserve the exact FP16 FMA loop, dense-lane layout, FP32 nonfused multiply
   and add, initial +0 semantics, full destination hazards, and finite queues.
   Implement a comparable direct CPU accumulation epilogue for private SPM
   (and cache). New same-ELF buffered/direct controls; bit-exact output gates.
2. Tune Bc at T4096 with fixed Br16/R256: 256/512/1024, independently for
   private SPM and L3. Existing Bc512 is reused, not rerun. The primary L3
   candidate uses FP16 contributions, first-update initialization, CPU final
   normalization, and the accepted serialized bank engine. No simulator change.
3. At each backend's selected Bc, test Br32/R256 and Br16/R128,R512, then
   compare paired versus sequential row submission in the direct epilogue.
   No queue/storage enlargement or new hardware pipelining. Keep original
   per-row update order. This is a bounded search, not global optimization.

Phase 1 includes full T1024 and T4096 checks and a one-entry reduction gate.
Each new Bc uses the independent rounded-GEMM/FP32-state reference. Same-Bc
kernel/schedule/group changes must retain exact output bits; different Bc may
change rounding, so each is independently validated against its own reference.
Copy, reduction, initialization, finalization, network-byte, config and finite
occupancy assertions remain mandatory. Failures stop new work; no tolerance
relaxation. Accepted subsets and failed evidence remain separate.

Decision criterion: lower complete FA ROI than the current 1133.008us T4096
L3 result AND an improved comparison against independently tuned private SPM.
Always retain stronger historical controls; do not infer benefit from traffic,
queue retries, or stalled-cycle sums. Direct-epilogue gains for the CPU control
must be reported too. No T16K or hardware area/energy conclusion is implied.

Resource preflight: 128 logical CPUs, 97.5% aggregate idle, 712GiB available,
zero swap use, 1.7TiB disk free, three existing simulations. At most TWO new
simulations (five total at launch), one compilation process, single-threaded
NumPy references. Before launching, require >=50% aggregate idle, >=128GiB
available RAM, >=100GiB disk free, and fewer than six live gem5 processes.
During each run retain memory/disk checks; excessive CPU load prevents further
launches. Existing SwiGLU/llama jobs and frozen dependencies are untouched.

Artifacts: `results/cacheflex_l3_spm/kernel_tuning_20260928`.
