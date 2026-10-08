# Near-L3 reduction experiment — 2026-09-28

Related ledger L3-01 and FA-32. Storage-only T1024 slows from117.3456us
private-SPM to173.4896us with shared-L3 output; network flits roughly double.
The new intervention moves the FP32 scale/add to the LLC bank, instead of
fetching old O to the CPU. No separate accumulator SRAM is added.

Preserved baseline: initial_20260928/build_v4 and guest_v4; active queue untouched.
New root: results/cacheflex_l3_spm/reduction_20260928. All attempts retained.

Initial design: four core-side source records plus four in-flight records/core,
four bounded sink records/bank, one active engine/bank. Payload256B plus scalar
header; real Garnet transfer, no dedicated free network.16 FP32 lanes,3cycle
multiply and3cycle add, separately rounded. Read35cycles, arithmetic9cycles,
write response35cycles; read/write each contend for common64B/cycle bank port.
These are explicit design assumptions, not calibrated RTL. Existing controller
TBEs bound metadata/result storage; local queue holds contributions. No new
coherence sharers for SPM. Synchronized single-writer handoff remains required.

CPU may resume after payload capture; pending per-slot records and barrier
credits last until remote installation. Sink locks cover all four vector slots.
No atomics, OS context switching, checkpoint or exception-flag support claim.

Plan: disabled exact controls; recurrence/RAW/WB/overwrite/release/handoff tests
for synchronous, asynchronous, and one-entry slow-resource modes; full-output
FA T1024/T4096 comparison to cache/private-SPM/L3-storage-only. Judge end-to-end
ROI, not summed request latency or traffic alone. No new result yet.

## Build and validation launch

Build_v1 completed, SHA4ea58941d571228e429e81b211794c14805c6e6141fe361872f20cd35884d23c.
Guest_v1 built; experiment_v1 coordinator3468627, at most two extra simulators.
Disabled exact cache/private/L3-storage controls run before new commands.
Six new source/ABI tests pass. The older unclaimed-read source test needed its
function-boundary extraction narrowed after new methods were inserted; its
zero-read requirement is unchanged.

The separate storage-only T4096 queue failed with vector range index12/size12;
cache1194.3868us and private1167.5820us accepted, L3 output not accepted.
This pre-existing candidate failure must be diagnosed before a valid T4096
shared/reduction comparison. It is not a slowdown measurement.

## Queue-size override caught by audit

Experiment_v1 finished a disabled private control but correctly failed config
acceptance: the generic finite-resource wrapper changed the new local queue
from4 to16. The engine-count bound stayed4, but queue provisioning was not the
declared configuration. Experiment_v2 explicitly overrides only that new queue
after the generic wrapper. Simulator/guest bytes unchanged; failed attempt kept.
Runner3791544 uses one simulator while debugger3791543 replays storage-only
T4096 to catch the bounds exception. At most two new simulations overall.
Source tests: six new plus eight shared-storage checks pass.

## Two diagnosed simulator defects; build_v2 / experiment_v3

Experiment_v2 accepted all three disabled controls with exactly the previous
non-host stats and full numerical outputs: cache120.9356us, private117.3456us,
shared output173.4896us. Its async directed test then deadlocked; no reduction
performance measurement was accepted. Trace replay is preserved in
reduction_20260928/diagnosis/smoke_async_deadlock. Contrary to the initial progress
interpretation, commands did execute: bank5 captured rows0x68500380/3c0, finished
row340 at41610400, then never started the remaining rows. Internal work ports
were checked BEFORE the expired bank-service timer was cleared, consuming the
last wakeup with no new event. Move timer clearing before those work ports;
retain all bandwidth/latency/buffer settings.

GDB replay of the older T4096 storage-only failure pinpointed TreePLRU::getVictim
via CacheMemory::cacheProbe. A physical16-way PLRU leaf indexed a compacted12-way
eligible-candidate vector after four ways became SPM. This could select a wrong
eligible victim before eventually throwing. New filtered selection retains
physical leaf indices, follows PLRU preference among eligible subtrees, and
does not mutate replacement state. Full candidate sets retain the original path.
Regression tests cover every8-way subset and physical16-way/12-candidate indices.
The actual gtest execution is pending; seven source/ABI and eight storage tests
pass. This repair may affect coherent replacements with claimed SPM ways; old
shared results are historical controls, not an unquestioned final baseline.

Build_v2 coordinator457777 freezes both repairs plus replacement source/tests.
Experiment_v3 coordinator572292 waits, then repeats disabled controls, three
directed gates and five-arm T1024/T4096 comparisons with two simulators maximum.
Failed artifacts, old binaries and unrelated jobs remain unchanged.

## Repaired validation passed; FA comparisons started

Build_v2 SHA dd86ce2d5ee53926e4e92e9f3dab620851f5c7ccd60e84a46e6c07f04802c27a.
Experiment_v3 passed all three disabled exact-stat/output controls. Async and
sync directed tests each completed1040 commands with zero errors, including
immediate final loads, cross-core reads after synchronization, writeback followed
by coherent loads, younger overwrite ordering, release and slot reuse. One-entry
buffers with40cycle arithmetic also passed1040 commands/zero errors. Directed
ROIs60.0364/60.0420/69.4792us respectively; these are correctness workloads,
not performance evidence for FA.

Native TreePLRU suite:24 tests from7 suites PASSED, including all8-way eligible
subsets and16-way tree with12eligible physical ways. Logs in build_v2/plru_tests_v3.
First attempt hit host assembler DWARF5 incompatibility; googletest now honors
existing CCFLAGS_EXTRA=-gdwarf-2. Second built objects but SCons skipped final
test link; third records/executes the exact link command, as the isolated simulator
builder already does. These test-tooling repairs do not change frozen simulator.

FA T1024 cache/private active, then shared-output/async/sync; T4096 follows.
scripts/analyze_l3_reduction.py writes accepted-only COMPARISON.md. No accepted
FA reduction result yet. Two extra simulator slots; unrelated jobs untouched.

## First accepted FA result (T1024)

Full independent output validation, exact prior output bits, configuration and
resource checks passed: new shared-ELF cache121.1296us/private118.6076us,
L3-storage171.0292us, L3-async151.2172us. Async completed8192/8192 commands;
source/in-flight maxima4 on every core, no engine request-admission retries.
Async is11.584% lower latency than matched L3-storage (~1.1310x), but remains
slower than the strongest original-ELF same-platform private117.3456us and
cache120.9356us controls. Do not report1.131x as speedup over standard CacheFlex.
Network flits710266->561441 with offload, versus344259 for original privateSPM;
this is supporting traffic evidence, not a causal additive runtime breakdown.
Synchronous FA still running, then T4096 queued. The accepted-result report
watcher1413047 refreshes COMPARISON.md every30s until queue completion/failure.

T1024 synchronous FA also accepted152.6612us with8192 completed commands and
full checks. Async saves0.946% relative to sync in this configuration; no
substantial overlap gain established. All five T1024 arms accepted. T4096 cache
and private controls now active (1757777/1758258 at launch); shared output,
async and sync remain queued. No T4096 shared/reduction result accepted yet.

## Reduction-only retry isolation (authorized follow-up)

Related L3-02 and FA-14: old copy-path isolation did not test this new reduction
capture queue. T1024 async recorded573256 source-full rejects, mapped into the
shared cacheBlocked flag. Source audit confirms this can gate unrelated coherent
requests. Retry counts alone do not establish removable latency.

New opt-in cacheFlexIndependentReduction (defaultfalse) routes ONLY bit60 SPMST
commands through a distinct logical Ruby request/retry interface into the SAME
sequencer/source buffer. All original port indices preserved. Same SPM store
issue budget, source/in-flight/sink capacities, NoC, bank service and arithmetic
latencies. Not an extra physical bandwidth path or accumulator SRAM. Dedicated
blocked flag is set on command rejection and cleared only by its retry callback;
that callback does not clear coherent/SPM/bridge backpressure. Existing dependency
tracking, SQ head ordering, installation acknowledgments and barriers unchanged.
New counters: isolated command rejects and accepted coherent requests while
the reduction interface is blocked. Source-buffer fullness may still stall
SPM SQ/dispatch; no claim of eliminating all pipeline coupling.

Build_v3 coordinator2255285. Queue scripts/run_l3_reduction_retry.py, retry_v1,
coordinator2588588, max2 additional sims. Same guest_v1 ELF. First default-off
T1024 cache/private/async exact nonhost stats and output gates; then isolated
async/sync/one-entry directed tests; then enabled cache/private no-op exact
controls; finally isolated async/sync T1024/T4096, compared to experiment_v3.
Exhaustive config delta audit allows only new flag/port wiring, not resource
changes. Full references, output hashes, copy/command counts and occupancy bounds
remain required.20 source tests pass; no runtime isolation result yet.
Original experiment_v3 and unrelated jobs/frozen dependencies left untouched.

Original experiment_v3 finished all T4096 arms while the retry build ran:
cache1202.3908/private1168.2400/L3-storage1305.9208/async1208.3864/sync1208.3864us.
All full numerical/hash/config/resource checks passed, including32768commands
for each reduction arm. Async and sync are exactly equal in ROI at this T.
Offload remains3.436% slower than matched privateSPM, but7.469% lower latency
than storage-only. The PLRU repair has now passed the former T4096 crash point
and completed the workload. Isolation targets this new path; no benefit assumed.

Build_v3 SHA b8642627e0218fda8281edde1978cdf0db1566443cce87418ebabbebcca9a927.
All3 default-off T1024 controls exactly match prior nonhost stats and outputs.
Isolated async/sync directed executions produced PASS/1040commands/zero errors,
but retry_v1 failed a host accounting assertion: reductionOrderingRetries also
counts dependent ordinary SPM loads/stores, not only capture rejections. Async
sourceFull55986 == isolatedRejects55986, orderingRetries6818; sync55981/55981,
orderingRetries5130. Coherent accepted requests while reduction-blocked159/150,
respectively, demonstrate actual independent progress. No numerical relaxation.

Corrected accounting checks isolated rejects between command-only SourceFull and
SourceFull+mixed OrderingRetries. Simulator/guests unchanged. Runner-at-failure
preserved; retry_v2 coordinator3213882 revalidates/reuses the3 accepted default
controls after checking prior hashes/configs/stats/full outputs, then repeats
directed cases and resumes no-op controls plus FA. No failed timing promoted to
acceptance; attempts preserved.20source tests still pass.

Retry_v2: all3 default exact controls reused after full revalidation; all3
isolated directed gates now ACCEPTED (async60.0132/sync60.0172/one-entry69.4592us,
1040commands each, zero errors). Coherent progress while blocked159/150/997.
Both enabled-interface/no-reduction controls exactly match every old finite
nonhost statistic and output: cache121.1296/private118.6076us. FA T1024 isolated
async/sync3342712/3342718 active at launch, T4096 pair queued. No FA timing result
for isolation yet; this proves functional decoupling, not its latency benefit.

## Retry-isolated T1024 FA accepted

Same ELF/resources: async151.2172->147.3932us (3.8240us,2.529% lower latency,
1.025944x); sync152.6612->147.2528us (3.543% lower). Full numerical/output-bit,
config/resource/copy/command checks pass. Coherent requests accepted while
reduction admission blocked126340 async/126556 sync, versus none tracked in
the coupled mode. Both enabled-interface/no-reduction controls remain exactly
unchanged. This establishes an end-to-end cost from the shared retry coupling;
it does not isolate every downstream effect or imply additive stall fractions.
Async still25.606% slower than strongest original privateSPM117.3456us.
T4096 isolated async3396757/sync3396915 active at launch; no results yet.
