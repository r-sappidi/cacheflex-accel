"""MOD-09: source-anchored feasibility bounds and accepted-only headroom evidence."""
import json,sys,time
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc');sys.path.insert(0,str(ROOT/'scripts'))
from build_access_timing import sha,save
from run_access_timing import stats
BASE=ROOT/'results/cacheflex_l3_spm/hardware_audit_20260928';DIRECT=ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2';BUILD=DIRECT.parent/'build_v4'

def main():
 BASE.mkdir(exist_ok=True);manifest=json.loads((BUILD/'frozen.json').read_text());evidence={}
 names=['mem/ruby/protocol/MESI_Three_Level_SPM-L2cache.sm','mem/ruby/system/SequencerReduction.cc','mem/ruby/system/Sequencer.hh','mem/ruby/network/garnet/NetworkInterface.cc','mem/ruby/network/garnet/GarnetNetwork.cc','mem/ruby/structures/CacheMemory.cc']
 for name in names:
  p=BUILD/'sources'/name;assert manifest[str(p)]==sha(p);assert sha(ROOT/'gem5/src'/name)==sha(p)
  evidence[str(p)]=sha(p)
 state=json.loads((DIRECT/'status.json').read_text());rows=[]
 for r in state['completed']:
  if not r['name'].startswith('t'):continue
  path=Path(r['path']);assert json.loads((path/'accepted.json').read_text())==r
  for name in ('accepted.json','stats.txt','config.json'):evidence[str(path/name)]=sha(path/name)
  s=stats(path)[0];cfg=json.loads((path/'config.json').read_text());total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
  bank_cycles=s['simTicks']/cfg['system']['ruby']['clk_domain']['clock'][0];core_cycles=s['simTicks']/cfg['system']['cpu_clk_domain']['clock'][0]
  commands=total('.reductionAccepted');banks=[]
  for i in range(8):
   p=f'system.ruby.l2_cntrl{i}.L2cache.';n=s[p+'l3ReductionStarts'];init=s[p+'l3ReductionInitializations'];b=cfg['system']['ruby'][f'l2_cntrl{i}']
   assert not b['l3_reduction_pipeline'] and b['l3_reduction_arithmetic']==9 and b['l3_reduction_convert']==6
   normal=n-init
   # Compulsory port cycles; exclude coherent/claim/release traffic. No latency attribution.
   port=normal*8+init*4
   # Timing-accounting floor for current serialized engine: read35+convert6+ALU9,
   # then4 port cycles before another start. Init skips35 read cycles.
   engine=normal*54+init*19
   banks.append(dict(bank=i,commands=n,initializations=init,compulsory_data_bytes=normal*512+init*256,
    compulsory_port_cycles=port,port_cycles_over_roi=port/bank_cycles,
    arithmetic_lane_issue_cycles=4*n,arithmetic_lane_issue_over_roi=4*n/bank_cycles,
    serial_engine_schedule_floor_cycles=engine,serial_engine_floor_over_roi=engine/bank_cycles,
    nominal_link_beats=9*n,nominal_link_raw_duty=9*n/bank_cycles,
    nominal_one_credit_floor_cycles=15*n,nominal_one_credit_floor_over_roi=15*n/bank_cycles))
  mean=total('.reductionLifetimeCycles')/commands if commands else None
  rows.append(dict(name=r['name'],roi_us=r['roi_us'],case=r['case'],commands=commands,banks=banks,
   mean_command_core_cycles=mean,mean_inflight_per_core_from_lifetime=total('.reductionLifetimeCycles')/core_cycles/8,
   source_full_retry_events=total('.reductionSourceFull'),spm_lsq_full_events=total('.spmLsqFullEvents'),
   max_spm_store_occupancy=max(s[f'system.cpu{i}.spmLsq0.boundedMaxStoreOccupancy'] for i in range(8)),
   max_spm_load_occupancy=max(s[f'system.cpu{i}.spmLsq0.boundedMaxLoadOccupancy'] for i in range(8)),
   noc_flits=total('.flits_injected::total'),direct_beats=total('.directBeats')))
 byname={r['name']:r for r in rows};effects=[]
 for t in (1024,2048,4096):
  d=byname.get(f't{t}_direct')
  if not d:continue
  for arm in ('noc','private','cache'):
   b=byname.get(f't{t}_{arm}')
   if b:effects.append(dict(t=t,baseline=arm,baseline_us=b['roi_us'],direct_us=d['roi_us'],speedup=b['roi_us']/d['roi_us'],latency_reduction_pct=100*(1-d['roi_us']/b['roi_us'])))
 save(BASE/'evidence.json',evidence)
 save(BASE/'analysis.json',dict(timestamp=time.time(),parent_stage=state['stage'],rows=rows,effects=effects,assumptions=dict(bank_cycle_ns=.5,cpu_cycle_ns=.4,normal_port_bytes=512,initialized_port_bytes=256,lanes_per_bank=16,banks=8,link_forward_bits=128,link_reverse_bits=128,links_data_wires_total=2048,source_contribution_storage_bytes=8192,flight_contribution_copies_in_cpp_bytes=8192,source_packet_half_payload_copies_in_cpp_bytes=4096,bank_result_data_max_bytes=8192,direct_assembly_bytes=2304),caveat='Resource duty calculations are isolated compulsory-work/accounting floors, not measured utilization or removable ROI fractions. Host C++ copies are not automatically distinct physical SRAM allocations. No RTL/PDK calibration or sensitivity simulations performed.'))
 lines=['# MOD-09: pre-RTL hardware feasibility and headroom audit','',f'Accepted evidence snapshot; parent stage: {state["stage"]}.','',
 '## Finding','',
 'The mechanism is architecturally plausible as a finite, bank-local vector update engine. Exact 2GHz timing, SRAM access organization, physical area/energy and routing cost remain unsubstantiated. Current evidence supports continued bounded modeling, not implementation signoff. No new simulator jobs or changes to frozen runs in this audit.','',
 '## Accepted performance','',
 '| T | Comparator | Direct speedup | Latency reduction |','|---|---|---:|---:|']
 for e in effects:lines.append(f"| {e['t']} | {e['baseline']} | {e['speedup']:.6f}x | {e['latency_reduction_pct']:.3f}% |")
 lines+=['','## Compulsory resource demand on the busiest bank','',
 '| Case | SRAM port demand / ROI | 16-lane ALU issue / ROI | Serialized engine schedule floor / ROI | Raw forward link beats / ROI | One-credit forward floor / ROI |','|---|---:|---:|---:|---:|---:|']
 for r in rows:
  if not r['name'].endswith('_direct'):continue
  keys=['port_cycles_over_roi','arithmetic_lane_issue_over_roi','serial_engine_floor_over_roi','nominal_link_raw_duty','nominal_one_credit_floor_over_roi']
  vals=[max(b[k] for b in r['banks'])*100 for k in keys]
  lines.append('| '+r['name']+' | '+' | '.join(f'{v:.3f}%' for v in vals)+' |')
 lines+=['',
 'These ratios exclude other traffic and queue waits. They are not measured total bank utilization, not parallelizable speedup bounds, and not proof that bursts cannot stall a core. All bank/engine/link costs overlap execution. A low arithmetic issue duty is evidence against average ALU throughput being the sole bottleneck; it does not rule out critical latency.','',
 '## Source-supported hardware contract','',
 '* Current Qwen cases use serialized engines (pipeline=false), initialization and FP16 transport. A 64-element row requires64 FP32 multiplies and64 adds for a normal update. A16-lane implementation needs four launch beats; separate3cycle multiply+3cycle add with four beats is consistent with a9cycle arithmetic assumption, but achievable clock remains unverified. There are eight engine replicas, not one engine shared by all cores.',
 '* SRAM: normal update256B read+256B write, init256B write only. l3Occupy charges4cycles at read and4 at write; ordinary requests/responses/unblocks also consume the shared port. This is a controller-level throughput constraint, not a validated SRAM macro organization, row decoder, ECC path or measured conflict model.',
 '* Timing: normal engine result timer is35read+6conversion+9arithmetic=50Ruby cycles; init15. Four write-port cycles follow. The35cycle response delay is attached to the queued completion after installation and does NOT keep the bank engine occupied. Describing85cycles as the serialized bank initiation interval would be wrong. Four source-flight credits remain live until completion returns.',
 '* Direct commands:9beats at16B/beat; TX1+wire2 gives arrival at12cycles, RX visible at13. Credit returns3cycles after delivery, so earliest nominal next launch is15cycles absent other scheduling/backpressure. Thus effective command throughput<=1/15cycles, not1/9;144B/7.5ns=19.2GB/s including header (payload17.07GB/s), versus32GB/s raw. Replies are1beat on a separate direction.',
 '* Endpoints are same-clock2GHz Ruby private-controller/LLC-bank NIs, already attached to the same router. This bypasses local arbitration, not CPU-to-bank multihop traffic. CPU2.5GHz to Ruby2GHz transfer still uses the existing modeled interface. Physical CPU-domain crossing and256B source capture wiring are not newly validated by the direct-link experiment.',
 '* Eight128bit direction pairs imply2048 data wires in aggregate before controls/clock/credits. These are eight local bundles, not one2048bit global bus. No area conclusion follows without distance, metal pitch, floorplan/congestion and buffering evidence. The return packets need only16B; a narrower return path is a possible area tradeoff but has not been measured.',
 '* Queues: four source records +four flights/core, four bank entries, one assembly slot/direction. Source capture arrays alone store8KiB globally. Flight structs retain another8KiB of256B contribution copies in C++, plus4KiB of FP16 Packet payload copies globally. C++ copies may be lifetime/ownership representations rather than distinct hardware arrays; a physical ownership map is required before claiming metadata-only flight storage or summing these as necessary hardware.',
 '* Bank result TBEs may retain up to8KiB of256B result data across32entries, plus queued input payload and pipeline registers. Determine which pre-existing TBE fields can actually be reused without overlapping live values. Direct assembly2.25KiB is additional transport buffering, not the entire reduction-engine cost. No accumulator SRAM is added: claimed O uses64KiB/bank atR256,512KiB total of the existing4MiB LLC.',
 '* Hazard scoreboard: at most8live source/flight commands/core x4lines =32distinct pending line keys/core. Logical boundedness is established; associative comparisons, issue-critical timing, barrier fan-in and arbitration fairness are not synthesized. Finite runtime stress is not a general protocol/liveness proof.','',
 '## Headroom interpretation and optimization priorities','',
 '* L3-09 direct versus NoC changes transport substantially more than complete FA runtime. It does not justify a major wire investment on performance evidence alone. Existing slow-link sensitivity atT1024 is retained; The remaining T4096 baseline repeats and slow-link sensitivity are pending; accepted direct/NoC results are included above.',
 '* L3-03 pipelining already gave near-null results on the older single-head kernels. Testing it on the current Qwen+init+direct configuration would be a revalidation with a materially changed guest/issue schedule, not a new mechanism. Sparse bank-wide arithmetic work and current gains make unlimited ALUs or raw-link widening low-priority guesses.',
 '* L3-04 finalization is already implemented and regressed; extra output writeback/coherent completion work was observed, but its separate runtime contribution was not isolated. Repeating it unchanged is not a new opportunity. A different output path would require a preserved coherence contract and a matched full-ROI comparison; do not simply remove ACKs or DRAM traffic.',
 '* Source backpressure/LSQ occupancy remains a candidate, not a proven removable runtime fraction. Peak48SPM stores also occurs in private-only arms. Isolated retries establish blockage events, not their critical-path cost. Existing retry isolation already improved older FA (L3-02 follow-up); a new issue-path intervention must preserve lane width, source buffer, memory ordering and barrier bounds.',
 '* AtBc256,d64, two GEMMs do65536FLOPs per row/tile versus128FP32 operations offloaded (about0.195% of that arithmetic count). The opportunity is CPU accumulator traffic/dependencies and overlap, not offloading most FA arithmetic. Realistic claims should measure instruction/traffic reduction and full ROI.',
 '* Next decision-changing diagnostic: on the frozen Qwen winner, independently reduce source-capture/credit delay and bank service latency, one mechanism at a time, keeping numerical ordering and finite resources. Use explicitly optimistic labeled diagnostic bounds first; if complete ROI scarcely improves even there, stop tuning that mechanism. This audit DOES NOT launch these cases or claim their results.',
 '* Before physical implementation, establish a credible parameter envelope from an actual cell library/published comparable FP32 pipelines and memory data. Sweep the supported envelope on matched controls. Shared hardware changes apply to cache/SPM too; do not grant only the proposed arm faster SRAM. Any double-speed diagnostic is a hypothetical sensitivity, not an automatically realistic technology point.','',
 '## Engine-size tradeoff worth testing before RTL','',
 'L3-02/03 modeled16lanes; narrower lane counts on the current Qwen kernel have not been measured. The low compulsory ALU issue demand makes right-sizing a more defensible question than adding ALUs. With unchanged3cycle multiply,3cycle add and3cycle conversion pipeline latencies, the same lane-beat accounting gives:',
 '', '| FP32/conversion lanes | Arithmetic result latency | Conversion result latency | Minimum arithmetic initiation interval |', '|---|---:|---:|---:|',
 '|16|9cycles|6cycles|4cycles|', '|8|13cycles|10cycles|8cycles|', '|4|21cycles|18cycles|16cycles|',
 '', 'These are conditional scheduling calculations, NOT characterized technology timings or area ratios. A narrower datapath may extend critical command latency even when average utilization is low. A future matched full-ROI sensitivity would decide whether a smaller engine preserves the measured benefit over private SPM. No such run is launched by this audit.', '',
 '## External evidence and remaining work','',
 '[CACTI](https://github.com/HewlettPackard/cacti) supports cache/memory area, access/cycle time and power estimates; no CACTI run or12nm calibration is claimed here. [HardFloat](https://www.jhauser.us/arithmetic/HardFloat.html) supplies IEEE floating-point RTL, not technology-independent timing or area. These are candidate tools, not substitute measurements.','',
 'Needed before an implementation claim: characterized FP32 lane/conversion timings; bank macro organization and shared-port/ECC schedule; full bounded-buffer/metadata ownership and area budget; plausible eight-local-link floorplan and clock crossing constraints. No new RTL, synthesis, energy, or end-to-end inference result is claimed.','',
 'Evidence files: analysis.json contains reproducible counts/formulas and accepted timings; evidence.json hashes the actual frozen build sources and accepted case artifacts. Source line references below refer to those snapshots:','',
 '* MESI_Three_Level_SPM-L2cache.sm:386 bank reservation,664 engine start,739 result installation,1128 shared ordinary traffic.',
 '* SequencerReduction.cc:61 capture,202 source-to-flight,241 completion; Sequencer.hh:269 physical-ownership audit starting point.',
 '* NetworkInterface.cc:387 direct serialization/credit; GarnetNetwork.cc:93 local endpoint restriction.']
 (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(rows=len(rows),effects=effects,report=str(BASE/'REPORT.md')),indent=2))
if __name__=='__main__':main()
