# FA-47 asynchronous PV at T1024

User requested useful CPU overlap after FA-46 established only blocking PV
service-time sensitivities. No existing accepted run tests this schedule.
Same Qwen-shaped causal FA: 14 Q heads, 2 KV heads, d64, 8 cores, R256,
Br16, Bc256, VL16, FP16 block PV and FP32 online recurrence.

Each core has at most one submitted PV command. It retains a 64-byte alpha
snapshot and output-row index, computes the next row block's QK/softmax/P pack,
then collects the previous PV result before submitting again. All pending PV
is drained before V restaging, normalization, or task reassignment. The engine
owns copied P and a bounded result buffer; CPU P scratch may be reused safely.
Blocking arms use the same helper and markers but retire immediately.

Six arms: ordinary cache/SPM, blocking/asynchronous 128/256-lane service budgets.
Every arm must pass T256 before T1024. No automatic T4096 promotion. Required:
full independent numerical oracle, exact prior and peer output bits, exact work,
copies, command/transport counts, config and LSQ bounds, pending-buffer checks,
zero blocking overlap and positive asynchronous useful-work overlap. Poll counts
and useful-work/service intersection ticks are recorded per core. Their sum
across cores is not an end-to-end runtime saving.

This is a functional arithmetic hook with enforced simulated result-ready time,
not an implemented asynchronous hardware engine: V reads remain functional;
service budgets are additive; bank/NoC arbitration is absent. CPU register
transport and original P/V read probes remain timed (some P traffic duplicates).
Overlap works in simulated time; host functional arithmetic is still serial.
Full-ROI async vs blocking isolates scheduling; comparison with ordinary SPM
measures net benefit only within this new guest/model. A valid speedup supports
further finite-resource modeling; a null result does not establish an upper bound.

Max four workers, two allowed CPUs each, shared atomic 20% CPU/RAM cap,
64 GiB extra headroom, initial 10 GiB lease each (8 GiB sim + 2 GiB runner),
admitted memory growth and instruction-progress watchdog. Llama left unchanged.
Sources, objects and guest isolated from all prior runs. Failed runs never reused
as accepted timings. See results/cacheflex_flash_attention/pv_async_20261005.
