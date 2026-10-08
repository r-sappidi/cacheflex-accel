FA-40 SPM-specific isolation series, user-authorized2026-10-04

Original winning matrix guest_v3, Br16/Bc256/R256,Hq14/Hkv2,d64 causal,8cores.
No guest or main simulator/source changes. Isolated lsq_unit.cc overlay only,
registers new observational counters; exact default nonhost stats/output gate.
Uses retained object set and the already implemented FA21 functional-copy oracle.

Independent interventions, always against unmodified SPM:
bridge96: copies use96 separate diagnostic metadata credits instead of sharing48
SPM store slots. bridge4096: high-budget idealization, not feasible-area claim.
Both retain every shadow record, original completion/retirement/RAW/WAR/DSB logic,
regular SQ/LQ48/32, real SPM stores48, and all downstream traffic and admission.
Only copy-shadow accounting in numFreeStoreEntries changes. Unknown/non-copy
shadows remain charged. Added counters expose actual total/copy record maxima and
sampled admission relief; they are not exposed stalls or additive runtime.

copy_request2: SPMCP L0 request delay4->2 matches ordinary request parameter only;
not equivalent complete fill latency or CPU cost. copy_oracle: existing functional
coherent reads to previously claimed slots, next-cycle ACK, first claims/aliases
still real; CPU instructions/translation/order preserved. Measure exercised share.
load_latency3: privateSPM response5->3 targets successful-send service7->5cycles;
verify measured service, not claim exact cache-hit equivalence. load_latency1 is
positive-floor idealization. load_ports2 matches cacheLoadPorts2; load_ports8 is
throughput headroom. Other backend bottlenecks remain. combined enables
bridge4096+oracle+response1+ports8, then compare against individual interventions.

38cases: exact private control; eight directed RAW/roundtrip/release gates for
bridge96/bridge4096/oracle/combined;12cases atT256 then12T1024 (cachecontrol and
combined cache no-op,10SPM arms); fiveT4096 cachecontrol/SPMcontrol/bridge4096/
oracle/combined confirmations. Cache combined must be exact allstats/bits no-op.
Full independent oracle/dense smallgate/exactpeerbits/percorework/copy completion/
config/finitebounds validation. No failed packed-producer code is used.

Existing FA39 staging-address simplification is reused as negative evidence, not
rerun or mislabeled as complete instruction-cost removal. This series retains
SPMCP instruction issue/retire costs. A fully free CPU submission model would be
a separate, wider idealization and is NOT silently approximated by service latency.
If combined gives substantial headroom (e.g.>=5% fullROI), individual changes and
their interaction identify candidates for finite implementation; if counter relief
has <=1% ROI gain, demote that constraint. Never equate counter magnitude with time.

History: FA21 transfer to current Qwen/protocol, not new idea; FA15–24 load/network
negatives remain; FA39 L2credits removed retries without benefit, motivating this
more targeted copy-shadow test. No duplicate activecases. Existing queues untouched.
One new simulator/two allowedCPUs, shared20%CPU/RAM cap,>=64GiBspare,10GiB initial
lease+admitted4GiB growth. Compile4GiB/link24GiB singlethread reservations.

Build prepare.py, compile.py, then link_run.py (foreground persistent session).
Root results/cacheflex_flash_attention/spm_isolation_20261004/{build_v1,experiment_v1}.
