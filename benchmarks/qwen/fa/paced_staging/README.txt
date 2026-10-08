FA-44 paced and overlapped V staging, retained-protocol Qwen FA.
User authorized implementation following FA42 packet blocking audit / FA43 null.
FA14 already tested async/doublebuffer on older singlehead shape and regressed;
this is targeted transfer to current Qwen, not a new overlap concept.
Same Br16 Bc256 R256, Hq14 Hkv2 d64 causal FP16, exact query/core assignment,
QK/PV/softmax order, V-only placement, copy bytes and final outputs.
No hardware timing/protocol modifications. All extra software overhead is timed.

Single guest modes:
cache: coherent; burst: original V-only stage-before-QK, 32KiB.
paced: stage16 V rows (32x64B copies) after each of the first16 softmax row
computations; full256row tile installed/pending before firstPV. No artificial
sleep or new fence; this moves copies later, which can lose existing overlap.
overlap: first tile uses paced startup. Two disjoint32KiB V slots in ways0/1,
sets0..255 and256..511. After each16query-row PV block issue16rows of nextVtile,
if that tile is within the task's causal extent. Sixteen blocks issue exactly
one nexttile. Alternate buffers each KVtile. Existing pending-load WAR/RAW
ordering retained; no coherent write to V and no new early slot reuse.
dbburst: same doublebuffer addresses/lifetime/release as overlap, but all512
copies issued together immediately before each tile's QK. This controls for
64KiB footprint/slot mapping. Overlap changes both startup and future-copy timing;
not a pure hardware/burstiness intervention and no additive runtime attribution.

All five modes runT256 fulloracle, thenT512 multi-tile/reuse gate, thenT1024 full
ROI. New same-ELF controls necessary for code/layout effects; historical valid
FA39 outputs reused for exact-bit validation at256/1024, not timing equivalence.
T512 full independent oracle and exactpeer outputs. Bounds/work/copycounts,
strictconfig/guest/argv, noL3commands, finiteLSQs, hashes and export checks retained.
Report compares paced/burst, overlap/dbburst, overlap/burst, and cache control.
Four host tests cover event-level lifetime/order/work, slotencoding, geometry,
and source hooks; fullgem5oracle remains required.

Five parallel arms10GiBlease each (8GiBsimAS+2GiBcoordinator), twoallowedCPUs each,
shared20%CPU/RAM and64GiBspare, controlledgrowth. No existing guests/queues edited.
Source and frozen parent hashes verified before launch. FullROI evidence must
show runtime AND network behavior change; aggregate means cannot prove burst
suppression. Follow-up bounded traces if timings motivate causal attribution.
