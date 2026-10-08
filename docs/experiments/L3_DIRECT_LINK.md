# Finite local core-to-L3 reduction transport (L3-09)

Related: L3-02 implemented bank-local accumulation and L3-03 tested pipelining
and 128-byte FP16 transport over Garnet. L3-04 tested initialization/finalization;
L3-05 removed temporary PV buffers. L3-06/07 established and tuned Qwen-shaped
causal GQA; L3-08 adds the missing independently selected cache candidates.
None bypassed the local Garnet NI/router for shared-L3 reduction commands.
The earlier near-private-SPM link is a different destination/engine experiment.

Decision: retain a direct transport only if accepted full FA ROI improves over
the best measured *same-kernel* NoC reduction control. Also report incremental
benefit over best accepted private SPM and independently tuned cache. A directed
microbenchmark improvement, lower flit count, or stall sum is insufficient.

## Hardware contract

* Eight fixed, local private-controller i <-> LLC-bank i connections. Both are
  already attached to router i; this removes local injection/router arbitration,
  not multihop traversal. Nonlocal reductions continue over Garnet.
* Default disabled. 128-bit data in each direction, Ruby clock 2 GHz (32 GB/s
  raw per direction), two cycles one-way registered-wire delay, one TX and one
  RX endpoint cycle. CPU clock remains 2.5 GHz. No zero-latency path.
* FP16 command 144 bytes / 9 beats; FP32 command 272 bytes / 17 beats.
  Completion 16 bytes / 1 beat. Headers, payload serialization, and finite
  reverse credit delay (RX endpoint + propagation) explicitly consume time.
* One RX assembly credit per direction/pair: 272 command bytes plus 16 reply
  bytes, total 2304 bytes across eight pairs, plus bounded metadata/control.
  Source protocol queue head remains occupied until delivery; no extra TX queue.
  Stop-and-wait credit discipline is deliberately conservative. Effective
  sustained packet bandwidth is below the raw link rate.
* Receive enqueue shares the existing NI protocol ejection port (one per cycle)
  and finite MessageBuffer. Backpressure never drops/overwrites a command.
  Pending state is one MsgPtr per endpoint, retained in the original source
  buffer for functional access and bounded by existing source/sink credits.
* Only op7 reduction command and completion use this transport. CPU capture,
  private-controller bridge, source issue budget, four source/four flight records,
  four bank sink entries, ALU timing, pipeline setting, 64 B/cycle shared bank
  service, 35-cycle bank timing, and physical LLC capacity are unchanged.
  Coherent claims/victim handling, other SPM accesses, output loads/writebacks,
  and release acknowledgments retain their original paths. No new accumulator
  SRAM, SRAM port, or reduction engine. Older same-address ordering remains
  governed by existing CPU/sequencer/controller hazards and runtime gates.
* Sensitivity: 64-bit transport and four-cycle wire delay, with identical engine
  and storage. Neither timing nor physical area is RTL/place-and-route calibrated.
  Links require endpoint/clock/control/wiring costs; 2304 bytes is only data
  buffering, not a total silicon area estimate. SE only; no checkpoint/FS claim.

## Validation and measurement

Preserve pre-change Garnet sources and build an isolated ELF/config tree.
Default-off cache/private/L3 Qwen T256 exact all finite nonhost stats and output
bits; enabled no-reduction cache/private must also be exact. Directed FP32,
mixed FP16/FP32, initialization/finalization, synchronous and one-entry cases
must retain zero arithmetic/coherence errors and all command/count bounds.
One-entry slow link tests backpressure; full Qwen T256 also checks math/RAW.

Every full FA arm rechecks all exported elements against the independent oracle,
exact prior bits, export hash, unchanged configuration except three direct-link
parameters, copy/command counts, finite LSQ/engine occupancy, and global guards.
Direct commands and completions must equal accepted reductions, wire bytes must
match their formats, beats must reflect configured width, and occupancy <=1.
Default-off arms must exactly reproduce prior simulator statistics.

Choose best accepted per-backend configuration from L3-08 when it completes:
Br16/Bc256/R128 at T1024 for private/L3, Br16/Bc256/R256 at T2048/4096.
Run same-new-ELF cache/private/NoC-L3/direct-L3 at T1024, T2048, T4096;
add narrow/slow direct sensitivity at T1024 and T4096. Keep all null/regressions.
No additional kernel search in this transport experiment.

At most one new simulator, six total. Validation and T1024 use the remaining
resource-guarded slot; T2048/4096 wait for L3-08 completion to freeze the final
cache choices. Each launch requires >=50% CPU idle, >=128 GiB
available RAM, >=100 GiB disk; ongoing >=64 GiB RAM and >=75 GiB disk.
Original active simulations and frozen inputs are untouched.

Evidence root: `results/cacheflex_l3_spm/direct_link_20260928`.
Build and experiment status files are authoritative; compiled is not validated.
