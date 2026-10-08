# Shared L3 scratchpad: initial SE prototype

Progress and decisions: [L3_SPM_PROGRESS.md](L3_SPM_PROGRESS.md).
This is **storage/access support only**. There is no reduction engine, DMA,
multicast, extra data backing, or performance-win assumption.

## Storage and interface

Physical L3 is Ruby's L2Cache controller. Eight banks each retain their existing
512KiB/16-way arrays; up to eight low ways per bank are eligible for lazy SPM
claim. A slot is removed from coherent replacement only when claimed. Its old
coherent occupant is evicted using existing recalls/writeback/acknowledgments.
Unused eligible slots remain cache-usable. At least two coherent ways must
remain; this initial address format exposes at most eight SPM ways.

Experimental architectural address: bit59 selects shared L3, bits22:20 select
bank, bits18:16 select way, bits15:6 select set. Vector chunks occupy successive
ways at the same set, as with existing private SPM. Ruby maps these addresses to
a separate simulator namespace (0x68000000 plus bank/way/set). Only aligned
1..256B operations, up to eight banks, and the current SE configuration are
supported. No full-system, tagged-address, privilege/isolation, or checkpoint
support is claimed.

Existing SPM instructions carry this address; no fake loads or functional
accesses substitute for timing transactions:

- SPMST lazily claims/zero-initializes slots and applies the vector byte mask.
- SPMLD returns data without creating coherent copies or sharer records.
  Unclaimed slots return zero, matching private SPM and allowing speculation.
- SPMCP obtains a fresh coherent snapshot using the existing private source
  protocol, then routes its installation to the selected LLC bank. Completion
  waits for installation. It does not occupy a private-SPM slot, but it is not
  a direct LLC-to-LLC DMA path: source data pass through the requesting private
  controller's existing TBE.
- SPMWB snapshots one 64B shared slot and uses the existing coherent overwrite
  protocol. It completes after coherent installation/ack, not snapshot capture.
- SPMREL drops a slot after software has drained all its users. It is not an
  implicit writeback; software must explicitly preserve needed contents.

One writer per region and explicitly synchronized publication/consumption are
the initial contract. The issuing core's existing LSQ/shadows provide local
ordering; cross-core synchronization must establish producer completion before
readers start and reader completion before reuse/release. No reduction atomics.

## Finite resources and timing

Requests and replies use existing finite message queues and Garnet resources.
128/256B payload types account for the full network transfer, not a single
64B line regardless of vector length. Copy/writeback source responses enter
existing bounded TBEs before waiting to issue the next request; no unbounded
new queue holds payloads. CPU-visible completion stays at the final ACK.

Initial shared-SPM LLC service delay matches the translated ordinary LLC's
35 CPU cycles. Do not assume a tag-bypass timing benefit. A separate matched
sensitivity probe doubles it to70 cycles.

The optional common service-port model serializes bank/controller service at
64B/cycle, charging wide SPM operations multiple cycles and ordinary controller
messages one cycle. This is a conservative abstraction including control
messages, **not** a detailed or calibrated SRAM arbitration model. New matched
performance arms enable it identically. Default-off controls leave it disabled
and must match existing statistics. No free additional SRAM ports are added.

## Gated evaluation

`scripts/build_l3_spm.py` builds an isolated simulator and snapshots inputs.
`benchmarks/cacheflex_flash_attention/l3_spm/build.py` builds isolated guests.
`scripts/run_l3_spm.py` limits this suite to two simultaneous simulators and
stops downstream work on failure. `scripts/analyze_l3_spm.py` reports only
accepted cases.

1. Exact old-binary, default-disabled cache/private-SPM counters and outputs.
2. Eight-core shared reads, predicated writes, immediate RAW, remote dirty
   source copy-in, coherent WB visibility, repeated release/reuse, and survival
   of ordinary dirty data displaced by claims; repeat with small finite FIFOs.
3. Dependent-load probes with one/eight active workers and service-latency
   sensitivity. These are not FA speedups or isolated raw SRAM latency.
4. FA T1024 and T4096: coherent cache, private-SPM V, and private-SPM V plus
   shared-L3 FP32 output residency. Arithmetic remains on cores. Each worker
   owns a separate shared-bank output region, initialized directly by stores.
   All FP16 outputs are checked against the independent rounded reference and
   against the existing accepted output hash. FA ROI includes initialization,
   staging, normalization, drains and release.

This FA adapter is deliberately a functionality/performance-sanity transfer,
not a best-to-best L3 kernel or evidence for a reduction mechanism. Existing
ledger FA-08 (coherent LLC retention) and FA-25–27/32 (private output residency/
reduction) are different mechanisms and must not be pooled with this suite.
