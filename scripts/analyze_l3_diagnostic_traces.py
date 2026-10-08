"""Read-only MOD-10 causal/PC/trace analysis; no simulation or model changes.

PC-region accounting uses verified instruction ranges for the frozen Qwen ELF.
Head residence is descriptive, not additive removable latency. Trace commands
are joined by (core, serial); uncompleted records are explicitly right-censored.
"""
import collections,hashlib,json,os,re,statistics,subprocess,sys
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path.insert(0,str(ROOT/'scripts'))
from run_access_timing import stats
from build_access_timing import sha,save
BASE=ROOT/'results/cacheflex_l3_spm/diagnostics_20260929/experiment_v1'
OUT=ROOT/'results/cacheflex_l3_spm/diagnostics_20260929/trace_analysis_v1'
ELF=ROOT/'results/cacheflex_l3_spm/qwen_gqa_20260928/guest_v4/guest.export-safe'
BUILD=BASE.parent/'build_v2'
PRIOR=ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2/runs'
REGIONS={
 'qk_main':(0x402c90,0x402f20),
 'softmax_main':(0x402f48,0x403198),
 'pv_reduction_inner_loop':(0x401ec0,0x401f0c),
 'reduction_submission_epilogue':(0x401f0c,0x40203c),
 'probability_packing':(0x402060,0x402644),
 'v_staging_loop':(0x403c64,0x403ca0),
 'normalization_main':(0x403408,0x4034bc),
 'scalar_expf':(0x40ef60,0x40f040),
}

def summary(values):
    values=sorted(values)
    return dict(n=len(values),mean=statistics.mean(values),min=values[0],
                median=statistics.median(values),p95=values[int(.95*(len(values)-1))],max=values[-1]) if values else None

def analyze_trace(path,s,command):
    start=int(next(x.split('=')[1] for x in command if x.startswith('--debug-start=')))
    end=int(next(x.split('=')[1] for x in command if x.startswith('--debug-end=')))
    roi_start=int(s['finalTick']-s['simTicks']);period=400
    events=[];commands={};samples=[];live=collections.Counter();previous={};integrals=collections.Counter()
    maxima=collections.Counter();progress=[]
    for line in path.read_text().splitlines():
        m=re.match(r'(\d+): system.ruby.l0_cntrl(\d+).sequencer: CFRED (.*)',line)
        if not m:
            p=re.match(r'(\d+): .*system.cpu(\d+) progress event, total committed:(\d+), progress insts committed: (\d+)',line)
            if p:progress.append(dict(tick=int(p[1]),core=int(p[2]),committed=int(p[3]),delta=int(p[4])))
            continue
        tick,core=int(m[1]),int(m[2]);data=dict(re.findall(r'(\w+)=(\S+)',m[3]));phase=data.pop('phase')
        event=dict(tick=tick,core=core,phase=phase,**{k:int(v,0) for k,v in data.items()});events.append(event)
        assert start<=tick<=end
        if phase=='sample':
            assert event['source']+event['flight']==live[core],('trace occupancy inconsistency',event,live[core])
            samples.append(event);continue
        key=(core,event['serial']);record=commands.setdefault(key,dict(core=core,serial=event['serial'],slot=event['slot']))
        assert record['slot']==event['slot'] and phase not in record
        record[phase]=tick
        if phase=='capture':
            integrals[core]+=live[core]*(tick-previous.get(core,start));previous[core]=tick;live[core]+=1
            assert live[core]==event['source']+event['flight']
            maxima[core]=max(maxima[core],live[core])
        elif phase=='complete':
            assert 'capture' in record and 'issue' in record
            integrals[core]+=live[core]*(tick-previous.get(core,start));previous[core]=tick;live[core]-=1
        if 'age' in event:assert event['age']==(tick-record['capture'])//period
    for core,n in live.items():integrals[core]+=n*(end-previous[core])
    paired=[r for r in commands.values() if 'complete' in r]
    per_core=[]
    for core in range(8):
        rows=[r for r in commands.values() if r['core']==core]
        if not rows:per_core.append(dict(core=core,captures=0));continue
        per_core.append(dict(core=core,captures=len(rows),issues=sum('issue' in r for r in rows),completions=sum('complete' in r for r in rows),
            first_capture_roi_us=(min(r['capture'] for r in rows)-roi_start)/1e6,
            serial_min=min(r['serial'] for r in rows),serial_max=max(r['serial'] for r in rows),
            outstanding_at_end=live[core],max_outstanding=maxima[core],
            average_outstanding_over_entire_trace_window=integrals[core]/(end-start)))
    first=min(e['tick'] for e in events if e['phase']=='capture')
    return dict(window_start_tick=start,window_end_tick=end,roi_start_tick=roi_start,window_us=(end-start)/1e6,
        roi_overlap_us=(end-max(start,roi_start))/1e6,first_capture_roi_us=(first-roi_start)/1e6,
        time_from_first_capture_to_window_end_us=(end-first)/1e6,
        roi_coverage_pct=100*(end-max(start,roi_start))/s['simTicks'],
        counts=dict(collections.Counter(e['phase'] for e in events)),per_core=per_core,
        paired_capture_to_issue_cycles=summary([(r['issue']-r['capture'])/period for r in paired]),
        paired_issue_to_complete_cycles=summary([(r['complete']-r['issue'])/period for r in paired]),
        paired_capture_to_complete_cycles=summary([(r['complete']-r['capture'])/period for r in paired]),
        right_censored_commands=len(commands)-len(paired),
        commands_coverage_pct=100*len(commands)/sum(v for k,v in s.items() if k.endswith('.reductionAccepted')),
        sampled_source_full=sum(e['source']==4 for e in samples),sampled_flight_full=sum(e['flight']==4 for e in samples),
        initial_only_by_guest_mapping=all(r['serial']<256 for r in commands.values()),
        progress_samples_in_trace_window=[p for p in progress if start<=p['tick']<=end],
        caveat='First 256 commands/core initialize the first query group in the frozen guest. This trace contains only that startup burst, not steady-state RMW or terminal drain. Service-conditional samples and censored lifetimes cannot estimate full-ROI stall fractions.')

def metrics(path):
    accepted=json.loads((path/'accepted.json').read_text());s=stats(path)[0]
    assert s['simTicks']/1e6==accepted['roi_us']
    total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
    count=total('.reductionAccepted');corecycles=s['simTicks']/400*8
    profiles={}
    for metric in ('pcHeadCycles','pcNoRetireCycles','pcSleepHeadCycles','pcRetiredMicroops'):
        pcs=collections.Counter()
        for k,v in s.items():
            m=re.fullmatch(r'system.cpu\d+.commit.'+metric+r'::(\d+)',k)
            if m:pcs[int(m[1])]+=v
        groups={name:sum(v for pc,v in pcs.items() if lo<=pc<hi) for name,(lo,hi) in REGIONS.items()}
        profiles[metric]=dict(total=sum(pcs.values()),regions=groups,top_pcs=[dict(pc=hex(pc),count=v) for pc,v in pcs.most_common(20)])
    for i in range(8):
        p=f'system.cpu{i}.commit.'
        assert s[p+'pcObservedCycles']==s[p+'pcHeadCycles::samples']+s[p+'pcEmptyRobCycles']
        assert abs(s[p+'pcObservedCycles']+s[p+'pcSleepCycles']-s['simTicks']/400)<100
    return dict(path=str(path),roi_us=accepted['roi_us'],case=accepted['case'],commands=count,
        initializations=total('.l3ReductionInitializations'),source_full_retries=total('.reductionSourceFull'),
        memory_retries=total('.reductionMemoryRetries'),ordering_retries=total('.reductionOrderingRetries'),
        mean_lifetime_corecycles=total('.reductionLifetimeCycles')/count if count else None,
        coherent_requests_while_blocked=total('.coherentSentWhileReductionBlocked'),spm_lsq_full_events=total('.spmLsqFullEvents'),
        instructions=s['simInsts'],copies=total('.m_spmcpRequestsIssued'),
        aggregate_corecycles=corecycles,profiles=profiles,output_sha256=accepted['output_sha256']),s

def main():
    budget=json.loads((ROOT/'results/parallel_budget_20260929/budget.json').read_text());os.sched_setaffinity(0,budget['cpu_set'])
    OUT.mkdir(exist_ok=True)
    names=['t4096_direct_trace','t4096_direct_read_fast','t4096_direct_reply_fast','t4096_direct_both_fast','t4096_direct_lanes4','t4096_direct_lanes8','t4096_direct_pipeline','t4096_noc_both_fast']
    result={};manifest={};baseline_stats=None
    for name in names:
        path=BASE/'runs'/name;result[name],s=metrics(path)
        if name==names[0]:baseline_stats=s
        for f in ('stats.txt','accepted.json','command.json','config.json'):manifest[str(path/f)]=sha(path/f)
    for arm in ('cache','private','noc'):
        path=PRIOR/f't4096_{arm}';result[arm],_=metrics(path)
        for f in ('stats.txt','accepted.json','command.json','config.json'):manifest[str(path/f)]=sha(path/f)
    base=result[names[0]]
    for name,row in result.items():
        row['latency_delta_pct_vs_direct']=100*(row['roi_us']/base['roi_us']-1)
        if row['commands']:
            assert row['commands']==base['commands']
            assert row['output_sha256']==base['output_sha256']
            row['retry_delta_pct_vs_direct']=100*(row['source_full_retries']/base['source_full_retries']-1)
            row['lifetime_delta_pct_vs_direct']=100*(row['mean_lifetime_corecycles']/base['mean_lifetime_corecycles']-1)
    trace_path=BASE/'runs/t4096_direct_trace/reduction_trace.log'
    trace=analyze_trace(trace_path,baseline_stats,json.loads((trace_path.parent/'command.json').read_text()))
    for p in (trace_path,ELF,ELF.parent/'driver.cpp',ELF.parent/'pv_direct.hpp',
              BUILD/'sources/mem/ruby/system/SequencerReduction.cc',BUILD/'sources/cpu/o3/commit.cc',Path(__file__)):
        manifest[str(p)]=sha(p)
    # Preserve disassembly so ranges and custom SPM opcode sites are reviewable.
    dis=subprocess.check_output(['aarch64-linux-gnu-objdump','-d','-C',str(ELF)],text=True)
    (OUT/'guest.disassembly.txt').write_text(dis)
    assert '401ed8:' in dis and '402d80:' in dis and '403c6c:' in dis
    save(OUT/'inputs.json',manifest);save(OUT/'analysis.json',dict(cases=result,trace=trace,regions={k:[hex(a),hex(b)] for k,(a,b) in REGIONS.items()},
        pc_caveat='Disjoint verified address regions, not whole semantic phases. Shares are aggregate ROB-head residency / eight-core elapsed cycles. No-retire is a subset and is not added. Posted stores may retire before downstream stalls appear at later PCs. Microops have a different denominator.'))
    lines=['# MOD-10: why faster L3 reductions barely change FA latency','',
      'Read-only analysis of accepted T4096 Qwen results; no new simulations or model changes. Related MOD-09/10, L3-02/03/05/09. All numeric values use the first full ROI, not post-ROI export.','',
      '## Causal result','',
      'Changing the engine sharply changes command lifetime/backpressure without materially changing full FA time. This supports latency overlap and a workload dominated by other CPU-side attention work; it does not establish that every source stall is harmless.','',
      '| Case | FA us | Mean command lifetime, core cycles | Source-full retries | FA change vs direct |','|---|---:|---:|---:|---:|']
    for name in names[:-1]:
        r=result[name];lines.append(f"| {name} | {r['roi_us']:.4f} | {r['mean_lifetime_corecycles']:.2f} | {r['source_full_retries']:.0f} | {r['latency_delta_pct_vs_direct']:+.4f}% |")
    lines+=['','Negative FA changes mean faster. Retry counts are attempts, not cycles; mean command lifetimes overlap across commands and cores. The approximately 0.1% shifts are not a proven reproducibility/noise floor. NoC both-fast is also regressive relative to its own NoC control.','',
      '## Full-run CPU location evidence','',
      '| Verified code region | Aggregate ROB-head residence / elapsed core cycles | No-retire subset / elapsed core cycles |','|---|---:|---:|']
    for region in REGIONS:
        a=base['profiles']['pcHeadCycles']['regions'][region];b=base['profiles']['pcNoRetireCycles']['regions'][region]
        lines.append(f"| {region} | {100*a/base['aggregate_corecycles']:.3f}% | {100*b/base['aggregate_corecycles']:.3f}% |")
    lines+=['','QK and PV regions include operand delivery and arithmetic, not pure compute time. The softmax range is its main contiguous block only; scalar expf is separate. Other branches and setup remain outside these ranges. Counts are summed across cores, not a serial phase-timing breakdown or additive speedup budget.',
      'The largest PV no-retirement site is 0x401ed8, the first FP16 FMLA consuming the V/P inputs. This identifies a dependency location, not whether its cause is V latency, P loads, FP execution, or shared issue pressure. QK hot sites0x402d80/0x402de0 are coherent K loads. Probability-packing hot site0x40252c and staging divide0x403c6c also remain CPU-side.',
      'The reduction epilogue is a small ROB-head region, but posted stores can retire before capture completes; its small no-retire share is not a standalone proof of zero admission cost. The controlled timing changes supply stronger causal evidence.','',
      '## Trace coverage and burst evidence','',
      f"ROI begins at tick {trace['roi_start_tick']}; the 100 us trace covers {trace['roi_overlap_us']:.4f} us of ROI ({trace['roi_coverage_pct']:.3f}%). First capture occurs {trace['first_capture_roi_us']:.4f} us into ROI, leaving only {trace['time_from_first_capture_to_window_end_us']:.4f} us of captured reduction activity.",
      f"There are {trace['counts']['capture']} captures, {trace['counts']['issue']} issues and {trace['counts']['complete']} completions; {trace['right_censored_commands']} commands are right-censored. This is {trace['commands_coverage_pct']:.5f}% of all {int(base['commands']):,} ROI commands.",
      'All captured serials are below256/core and start at0. The frozen driver first initializes256 output rows/core, so this window observes initial-contribution bursts, not steady-state read/modify/write. Its high source/flight-full sample fractions must not be generalized to full FA. In particular, this trace cannot explain the normal accumulator-read latency distribution.',
      f"Across the complete run, {int(base['coherent_requests_while_blocked'])} regular requests were accepted while reduction capture awaited retry: retry isolation permits continued useful memory progress. This is an event count, not proof that all those requests retire off the critical path.",'',
      '## Scope of the incremental gain','',
      f"Private-SPM {result['private']['roi_us']:.4f} us -> direct-L3 {base['roi_us']:.4f} us is {100*(1-base['roi_us']/result['private']['roi_us']):.3f}% lower latency. NoC-L3 {result['noc']['roi_us']:.4f} us -> direct-L3 saves {100*(1-base['roi_us']/result['noc']['roi_us']):.3f}%. The overall gain over cache includes private-SPM benefit and must not be attributed wholly to reduction.",
      'For a full Bc256,d64 row/tile, QK+PV perform65536 FLOPs; the normal L3 update performs128 FP32 multiply/add operations (about0.195% of that arithmetic). Initialization omits the read and multiply. Benefits therefore come mainly from avoiding CPU output-accumulator traffic/dependencies, not moving most FA arithmetic off the core. CPU QK/PV, softmax, packing, V staging and final normalization remain.',
      'Direct PV epilogues already removed the buffered PV-output store/reload (L3-05); retry isolation already removed global coherent backpressure coupling (L3-02). Re-proposing either as unimplemented would duplicate existing work.','',
      '## Decision and remaining uncertainty','',
      'Current evidence does not support source queue size, faster SRAM/replies, additional ALUs or direct-link bandwidth as a major next throughput investment for this kernel. A smaller engine is supported as a performance-preserving candidate; area/timing feasibility is still uncharacterized.',
      'The evidence identifies CPU-side QK/PV operand delivery, softmax and probability packing as candidates for software optimization; no particular removable fraction is established. A common optimization must be applied to cache/private/L3 controls, so it may improve total FA without increasing incremental L3 benefit. The direct softmax-to-packed-P idea remains distinct from the already implemented direct PV epilogue (FA-06/29); this analysis does not implement or launch it.',
      'To attribute a residual submission blockage, a follow-up would need steady-state and drain windows plus commit/issue correlation and an exact instrumentation-equivalence gate. That is materially new temporal coverage, not another read/reply sweep. Its decision criterion remains lower full FA ROI from a bounded intervention; window stalls alone do not establish gain. Long-context conclusions await the existing T16K/T32K cases.',
      '', 'Reproduction: python3 scripts/analyze_l3_diagnostic_traces.py. analysis.json contains per-case counters/PCs and paired per-command trace statistics; inputs.json hashes all source evidence.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT/'REPORT.md'),trace={k:trace[k] for k in ('roi_overlap_us','first_capture_roi_us','time_from_first_capture_to_window_end_us','counts','commands_coverage_pct','initial_only_by_guest_mapping')},counter_deltas={n:{k:result[n][k] for k in ('latency_delta_pct_vs_direct','retry_delta_pct_vs_direct','lifetime_delta_pct_vs_direct')} for n in names[:-1]}),indent=2))

if __name__=='__main__':main()
