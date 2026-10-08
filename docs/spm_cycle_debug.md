# Cycle-level CacheFlex debugging

Do not begin with an eight-core W7 `ExecAll` trace. It is too large to align
SPM copy, shadow-table, Ruby, and pipeline events reliably. Use the same VL=16
kernel with one core, one 24-by-128 output tile, and a short K first; then grow
K to 256 and finally return to eight cores.

## Recommended trace ladder

1. Run `M=24, K=16, N=128, KC=16` with `SPM_CPU`. This exposes SPMCP issue,
   shadow-table insertion/blocking/wakeup, SPMLD issue, and completion with a
   tick on every line.
2. Add `RubySequencer,RubyProtocol,RubyResourceStalls` to follow each copy from
   the CPU port through Ruby and back.
3. Add `LSQ,LSQUnit,IEW,Commit` only for a narrow tick interval around one K
   iteration. This identifies whether the oldest blocked instruction is in the
   shadow table, LSQ, issue queue, or commit path.
4. Use `O3PipeView` for that same interval and render it with
   `gem5/util/o3-pipeview.py`. It gives the clearest instruction-by-instruction
   fetch-to-commit timeline.
5. Add `RubyNetwork` only after a single-core trace is understood. For the
   eight-core case, prefer counters and filtered protocol events because full
   Garnet tracing grows extremely quickly.

Useful gem5 options are:

```text
--debug-flags=SPM_CPU,RubySequencer,RubyProtocol,RubyResourceStalls
--debug-file=spm.trace.gz
--debug-start=<first-interesting-tick>
--debug-end=<last-interesting-tick>
```

For an instruction trace, use:

```text
--debug-flags=Exec,ExecCPSeq,ExecEffAddr,ExecSymbol,SPM_CPU
```

For a pipeline trace, use:

```text
--debug-flags=O3PipeView,SPM_CPU
```

First run with only `SPM_CPU`, find the first SPMCP tick, and rerun with the
larger flag set and a bounded `--debug-start/--debug-end` interval. This avoids
tracing static-binary startup and pthread initialization.

## Events to align

For one destination set, record:

1. SPMCP decode/rename/LSQ insertion sequence number.
2. Request accepted or retried at the SPM port.
3. Ruby request issue and source-line hit/miss.
4. SPM installation and response tick.
5. Shadow-table copy completion and dependent-load wakeup.
6. SPMLD issue, response, writeback, and dependent FMLA issue.
7. ROB-head and commit behavior during the interval.

The bottleneck classification follows directly:

- Long issue-to-Ruby-accept gap: SPM port, LSQ, or Ruby backpressure.
- Long Ruby-accept-to-response gap: cache/NoC/directory latency.
- Response returned but SPMLD remains blocked: shadow-table wakeup/replay.
- SPMLD completes but FMLAs issue late: IQ/dependency/FU pressure.
- Backend progresses but retirement stalls: ROB-head or ordering constraint.

For multicore analysis, correlate this trace with per-core SPMCP busy cycles,
outstanding-cycle area, maximum outstanding copies, Ruby resource stalls,
Garnet packets/flits/hops, and DRAM traffic. Cycle traces explain one event;
counters establish whether that event dominates the complete W7 run.

## True debugger stepping

If a trace reveals a suspicious sequence number, add a temporary conditional
call to `debug::breakpoint()` at that sequence/tick and run gem5 under GDB.
This is useful for inspecting the shadow table and LSQ state, but it is slower
than trace-driven analysis and should be used only after narrowing the event.
