FA-37 DSB observer-only diagnostic (2026-10-03)

Question: does DSB wait materially for unrelated coherent operations in FA-36?
FA-34/36 establish cyclic schedule cost, not its attribution. No sync changes.
Main/frozen simulator files untouched; four private replacement .o files plus
existing accepted optimistic ring pseudo_inst.o link against original objects.
No hardware queues/storage/timing/guest/ordering changes. CACHEFLEX_DSB_DIAG
enables collection; new statistics exist at zero when disabled. Source snapshots,
compile commands, original source, link inputs and full manifests retained.

Per-CPU dsbDiag counters:
 arrivals: unique unexecuted DSB at ROB head; retired: architectural DSB count.
 headSamples: distinct CPU ticks with a DSB commit attempt.
 widthDeferredSamples: attempt after an earlier instruction committed that cycle.
 storeWaitSamples: attempts at first commit slot blocked by existing drain rule.
 waitMask[0..7]: mutually exclusive mask coherent=1, SPM=2, bridge=4. Mask1 is
 coherent-only drain waiting. Mask0 is unknown and must be reported, not discarded.
 Bridges identified separately even though their instruction lives in regular SQ.
 Outstanding posted reductions count as SPM, not coherent. Shadow/SQ snapshots
 are deduplicated by instruction sequence for arrivalRequests. Counts are LSQ
 operations, not network packets. Older loads already retired before the DSB can
 reach ROB head normally do not appear; this is not a complete dispatch snapshot.
 younger*Dependencies: speculative memory instructions assigned an existing DSB
 dependency in the memory dependence unit, not proof DSB was their sole blocker.
 renameWaitSamples: serialization samples attributed to the preceding DSB.
 renameCoherentFrontSamples/Unique: visible ordinary coherent memrefs at the
 blocked rename front. Does not count all hidden younger instructions. DSB's
 serialize-after behavior may make MemDep counters zero; that is meaningful.

Counts are sampled at existing pipeline checks, deduplicated per tick, not an
additive elapsed-runtime decomposition. Rename samples overlap drain samples.
Sum across cores measures core waiting, not wall-clock saving. Coherent-only
wait is a candidate opportunity, not a predicted SPM-sync speedup. Synchronization
would still require SPM/bridge completion and arrival coordination.

Replay: two observer-OFF small controls then observer-ON nine existing FA-36
cases (T1024Q2, T1024Q14, T4096Q14, modes1/16/17). Same frozen guest and model.
Each requires exact ALL pre-existing finite numeric non-host stats in every
epoch, exact output bytes and normalized config, and independent full saved
oracle validation. Check vector sum identity and disabled zeros. A mismatch
halts downstream work. Directed classification semantics are not a full proof;
interpret measured category counters subject to documented pipeline scope.

One new simulator,8GiB lease (6GiB initial simulator AS+2GiB host), admitted
4GiB growth; link24GiB lease/23GiBAS, one thread. Shared20% CPU/RAM and64GiB
host headroom. Namespace PIDs are local; use status/log heartbeat, not host kill.
