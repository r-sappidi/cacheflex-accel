# FA-48: diagnose asynchronous PV overhead at T1024

FA-47 established correct asynchronous execution but a 5.416% net slowdown versus
same-guest ordinary SPM. FA-46 was blocking and cannot attribute the residual.
This study preserves arithmetic, ownership, R256/Br16/Bc256 and finite256 service,
and isolates the following costs within a new, matched guest.

First run ordinary SPM and asynchronous PV with passive PC observation OFF/ON.
ON must produce EXACTLY identical non-host statistics, configuration and output
bits to OFF before interventions launch. Assembly labels add no guest instructions;
compiler layout may differ from FA-47, so absolute old/new timings are not causal
comparisons. PC events observe commit boundaries without modifying architectural
state or scheduling events. Per-core phase intervals partition workbegin->workend.
QK/softmax/packing entry counts must match work; every timeline must sum exactly.

Then run five interventions: remove redundant timed CPU P/V read probe (flag1),
zero engine service time (flag2), both (3), remove redundant outer DSB/ISB pairs
before input and after collection (4), and remove probe plus outer fences (5).
All chunk-level register handoff fences and ISBs remain. Actual P data still
traverses CPU loads/registers; V remains claimed until completion. No engine
oracle outputs/precomputed values are substituted. Zero service is an optimistic
sensitivity only. Removing the probe also removes its CPU instructions/traffic:
this is not an isolated network-latency experiment. Removing only outer fences
is not a test of eliminating all synchronization costs.

Phases: 0 startup;1 K pack;2 pack barrier;3 Q pack;4 V staging/control;
5 QK;6 softmax;7 P pack;8 completion wait;9 result transfer;10 outer result fence;
11 recurrence;12 retirement bookkeeping;13 operand probe;14 outer input fence;
15 input call;16 input begin;17 input payload/chunk barriers;18 reserved;
19 command submit;20 loop control;21 CPU PV/recurrence;22 normalize;
23 release;24 final barrier;25 reporting. The same PC can fire for successive
micro-ops; acceptance requires correct primary-phase work counts. Phase boundaries
are retirement observations, not attribution of every overlapped micro-op.
Compare critical-core timelines and full ROI, never sum multicore ticks as runtime.

All nine arms pass T256 before T1024 (18 cases); full independent oracle,
historical/peer exact bits, copies/work/transport/service/config/LSQ/lifetime and
overlap checks retained. Zero-service must have zero blocked polls; no-probe
must produce no CPU SPM load responses. No T4096 promotion or unrelated tuning.
Five workers max, each two allowed host CPUs and initial10GiB lease with admitted
growth; shared20%CPU/RAM and64GiB extra headroom. Llama unchanged. Existing valid
FA46/47 sources/results untouched. Build/link isolated.

Decision: does removing the probe or unhidden service recover the net gap?
Do common QK/softmax intervals increase, or do explicit transfer/collection costs
exceed the CPU PV time saved? Factorial differences reveal interaction; phase
observations locate costs but do not independently prove their causal origin.
Finite bank/NoC contention and physical engine hardware remain unmodeled.

Final outcome: all22 cases (including four inert-startup controls in sibling
pv_diag_control) passed. Initial flag0 versus nonzero comparisons had a measurable
startup confound; use the final inert comparisons in REPORT.txt. Probe removal
saves106.176us, zero service19.172us, outer fences7.5984us. Finite no-probe/fewer-
outer-fences PV is913.4156us versus950.7684us ordinary SPM (3.929% lower), within
this diagnostic model only. All cases complete; no automatic target promotion.
