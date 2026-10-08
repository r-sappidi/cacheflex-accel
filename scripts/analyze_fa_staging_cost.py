"""Read-only transaction-path reconstruction and whole-ROI counter analysis."""
import collections
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
B = ROOT/'results/cacheflex_flash_attention'


def first_stats(path):
    section=path.read_text().split('---------- Begin Simulation Statistics')[1]
    out={}
    for line in section.splitlines():
        fields=line.split()
        if len(fields)>1:
            try: out[fields[0]]=float(fields[1])
            except ValueError: pass
    return out


def main():
    trace=B/'staging_trace_20260917/trace/diagnostic.trace'
    events=collections.defaultdict(list)
    pattern=re.compile(rb'^\s*(\d+)\s+(\d+)\s+(Seq|L0Cache|L1Cache|L2Cache)\s+(\S+)\s+(\S+)\s+\[0x[0-9a-f]+, line (0x[0-9a-f]+)\]\s*(.*)')
    with trace.open('rb') as stream:
        for line in stream:
            m=pattern.match(line)
            if m:
                tick,core,ctrl,event,state,addr,detail=m.groups()
                if core==b'0' or ctrl==b'L2Cache':
                    events[addr.decode()].append(dict(tick=int(tick),core=int(core),controller=ctrl.decode(),
                        event=event.decode(),state=state.decode(),detail=detail.decode(errors='replace')))
    paths=[]
    for addr,ev in events.items():
        begins=[e for e in ev if e['controller']=='Seq' and e['event']=='Begin' and e['detail'].strip()=='SPMCP_fetch']
        for start in begins:
            done=next((e for e in ev if e['controller']=='Seq' and e['event']=='Done' and e['tick']>=start['tick']),None)
            if done is None: continue
            middle=[e for e in ev if start['tick']<=e['tick']<=done['tick']]
            priv=[e for e in middle if e['controller']=='L1Cache' and e['event']=='SPMCP_fetch' and e['state']=='I>IX_D']
            llc=[e for e in middle if e['controller']=='L2Cache' and e['event']=='SPM_GETS_SILENT' and e['state']=='M>M']
            fill=[e for e in middle if e['controller']=='L1Cache' and e['event']=='Data_all_Acks' and e['state']=='IX_D>X']
            # Reject ambiguous repeated/other-request paths rather than guessing provenance.
            if len(priv)!=1 or len(llc)!=1 or len(fill)!=1: continue
            ticks=[start['tick'],priv[0]['tick'],llc[0]['tick'],fill[0]['tick'],done['tick']]
            if ticks!=sorted(ticks): continue
            values=[(b-a)/400 for a,b in zip(ticks,ticks[1:])]
            paths.append(dict(address=addr,ticks=ticks,total_cycles=(ticks[-1]-ticks[0])/400,
                intervals_cycles=dict(zip(['admission_to_private','private_to_llc','llc_to_install','install_to_done'],values))))
    stats=first_stats(B/'llc_retain_20260915/runs/spm_t16384_retain1_small0_old0/stats.txt')
    def total(suffix):return sum(v for k,v in stats.items() if re.fullmatch(r'system\.ruby\.l0_cntrl\d+\.sequencer\.'+suffix,k))
    count=total('m_spmcpRequestsIssued'); area=total('m_spmcpOutstandingCycleArea'); busy=total('m_spmcpBusyCycles')
    assert count==total('m_spmcpRequestsCompleted')
    assert total('m_spmcpAliasedRequests')==0
    assert stats['system.ruby.L2Cache_Controller.M.SPM_GETS_SILENT::total']+stats['system.ruby.L2Cache_Controller.NP.SPM_GETS_SILENT::total']==count
    for path in paths:
        assert sum(path['intervals_cycles'].values())==path['total_cycles']
    results=dict(scope='Old retained T16K/Hq8/Hkv1/512KiB private L2, not current MICRO single-core runs',
        copied_lines=count,copied_bytes=count*64,mean_request_cycles=area/count,
        mean_outstanding_while_busy=area/busy,mean_core_copy_busy_ms=busy/8/2.5e6,
        amortized_busy_cycles_per_copy=busy/count,
        private_I_fetch=stats['system.ruby.L1Cache_Controller.I.SPMCP_fetch::total'],
        llc_resident_responses=stats['system.ruby.L2Cache_Controller.M.SPM_GETS_SILENT::total'],
        llc_memory_fetches=stats['system.ruby.L2Cache_Controller.NP.SPM_GETS_SILENT::total'],
        trace_complete_unambiguous_llc_hit_paths=len(paths),trace_paths=paths)
    if paths:
        results['trace_means']={k:statistics.mean(p['intervals_cycles'][k] for p in paths) for k in paths[0]['intervals_cycles']}
        results['trace_mean_total_cycles']=statistics.mean(p['total_cycles'] for p in paths)
    results['single_head_network_width']={}
    for arch in ('cache','spm'):
        rows={}
        for name in ('control','network256'):
            p=B/f'contention_optimization_v2_20260920/runs/{arch}_t16384_{name}'
            accepted=json.loads((p/'accepted.json').read_text())
            c=accepted['counters']
            rows[name]={k:c[k] for k in ('roi_us','m_spmcpBusyCycles','m_spmcpRequestsIssued','average_packet_network_latency','average_packet_queueing_latency','regular_load_mean_cycles')}
        rows['runtime_reduction_percent']=100*(1-rows['network256']['roi_us']/rows['control']['roi_us'])
        results['single_head_network_width'][arch]=rows
    results['caveats']=['Trace deliberately selected around worst staging gap; not representative latency average',
        'Controller-to-controller intervals combine modeled service, queues, arbitration and transport',
        'Busy-time/PC attribution is not removable end-to-end latency',
        'Network experiment uses different single-head hardware and changes width for all traffic, not SPMCP alone']
    if '--save' in sys.argv:
        output=B/'staging_cost_audit_20260921'
        output.mkdir(exist_ok=True)
        (output/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k!='trace_paths'},indent=2))


if __name__=='__main__':main()
