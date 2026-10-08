FA46 v5 repair. All prior versions/accepted controls preserved separately.
Same T256->T1024 shapes, percore work/copies, fulloracle/exacthistoricalbits,
configuration/resource checks; same SoftFloat sequentialFP16 PV and CPU FP32
recurrence. No relaxed acceptance, original numericalfailures remain failures.

Replaces ALL coherent functional reads/writes with explicit register transport:
CPU loads four256B SVE registers, serializing diagnostic captures1024B;
output returned through samefourregisters and real CPU stores. Ordered chunk
counts/bytebounds validated; one outstanding command percore;16KiB input/output
host buffers are diagnostic state, not claimed physicalengine SRAM. Existing
SPM V read remains functional, preclaimed/drained, sameoriginal hook method.
All payload values come from actual CPU accesses, no direct memory/answer oracle.
Bulk register transfer changes instructions/traffic/overhead fromv4. Same-newELF
cache/SPM controls required; don't pool old timings or call this a hardware bound.
PVoracle now includes P/result CPUtransport; PVtraffic adds original paired
P/Vloadprobe (P traffic duplicated intentionally, conservative sensitivity).
Packing gets real contiguousCPUreads/stores, command overhead retained.

Finite service uses readyTick and CPU polling; no quiesce/suspend of O3thread.
Original service budgets5472/3424/2400/6152cycles retained, additive aftertransfer.
Poll/transfer overhead is timed. No asynchronousengine/NoC/port/queue simulation.
No inference about realL3performance. Commandguard checks output ready, fullinput,
ordered/nonoverlapping transfers; complete stats verify exactbytes andcommands.

Runner checks committed-instruction progress every5s, fails after120hostseconds
without progress (pressurepauses exempt),30minT256/90minT1024 walllimits. This
prevents liveheartbeat/noCPUprogress being mistaken for productive execution.
Shared20%CPU/RAM cap,64GiBhostheadroom, max4workers10GiBlease each, allowedCPUset.
NoT4096automaticlaunch in this monitored repair; report and reviewT1024first.
