# SPM release versus coherent GETX regression

`spm_release_getx.cpp` runs two pthreads over 64 shared 64-byte lines for
64 iterations. One thread writes known data and copies each line to its private
SPM; the other concurrently stores a different value in the first word.
Explicit assembly shared-data accesses avoid a C++ non-atomic data race.
Barriers delimit iterations, not the overlapping copy/store operations.

Each SPM snapshot must contain either the old or new first word and unchanged
remaining words. After the writer finishes, coherent reads must retain every
new first word. Thus merely breaking the deadlock while dropping the new
owner's stores does not pass. This is a race stress test, not an exhaustive
protocol proof or a performance benchmark.

Build in a fresh directory with `build_release_getx.sh OUTPUT`. The runner
`scripts/run_spm_release_getx.py OUTPUT --simulator FIXED_GEM5` compares the
preserved pre-fix simulator with the repaired one using identical VL4 guests
and configuration. Protocol traces must show actual MX_A -> IX_A transfers
and LLC SPM_PUT_old acknowledgments; a successful run without those transitions
does not count as coverage. Output includes command JSON, logs and result JSON.

The original simulator deadlocks. Initial repair exposed a second race:
GETX stalls during MX_L0 recall, but is never awakened when recall completes.
The repair therefore includes wakeups on SPM source-recall completion, as well
as existing-TBE snapshot forwarding and old-owner release handling at the LLC.

The artificial request-latency-50 experiment in
`results/ruby_getx_stress_20260916` hit a separate FIFO-ordering assertion in
both builds and is not evidence for this fix. Default-latency tests are used.
The pre-fix and failed intermediate results are retained for diagnosis.
