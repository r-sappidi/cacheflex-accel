"""Independent aggregate/occupancy cross-check and concise completed-run report."""
import json
from pathlib import Path
from build_access_timing import ROOT, save
from run_access_timing import stats
from run_access_aggregate import validate_aggregate

def occupancy_check(path):
    data=json.loads((path/'aggregate.json').read_text())
    counters=stats(path)[0]
    expected=int(sum(v for k,v in counters.items() if k.endswith('.m_spmcpRequestsIssued')))
    validate_aggregate(data,expected)
    checks={}
    for core in range(8):
        g=data['groups'][f'core_{core}']
        pre_ruby=sum(v for k,v in g['category_ticks'].items() if k.startswith('CPU_'))
        observed=g['total_ticks']+g['ack_after_install_ticks']-pre_ruby
        area=counters[f'system.ruby.l0_cntrl{core}.sequencer.m_spmcpOutstandingCycleArea']
        assert observed==area*400,(core,observed,area)
        checks[str(core)]=dict(aggregate_ticks=observed,independent_outstanding_area_ticks=int(area*400))
    save(path/'occupancy_crosscheck.json',dict(exact_percore_match=True,cores=checks))
    return data

def main():
    root=ROOT/'results/cacheflex_flash_attention/access_aggregate_20260922'
    ex=root/'experiment_v3'
    assert json.loads((ex/'status.json').read_text())['stage']=='completed_validated'
    occupancy_check(ex/'spm_t512_on')
    data=occupancy_check(ex/'spm_t4096');g=data['groups']['all']
    cache=json.loads((ex/'cache_t4096/validated.json').read_text())['roi_us']
    spm=json.loads((ex/'spm_t4096/validated.json').read_text())['roi_us']
    buckets={}
    for k,v in g['category_ticks'].items():
        if k.startswith('CPU_'):name='CPU issue/commit/admission'
        elif k=='L1_ingress_FIFO_wait':name='L1 ingress FIFO waiting'
        elif k.startswith('L1_forward'):name='L1 forwarding resource waits'
        elif k.startswith('network_'):name='Network transport/queueing'
        else:name='Local/controller/remaining service'
        buckets[name]=buckets.get(name,0)+v
    summary=dict(copies=g['count'],mean_copy_cycles=g['total_ticks']/g['count']/400,
                 cache_roi_us=cache,spm_roi_us=spm,speedup=cache/spm,
                 aggregate_bytes=(ex/'spm_t4096/aggregate.json').stat().st_size,
                 coarse_categories={k:dict(mean_cycles=v/g['count']/400,percent=100*v/g['total_ticks']) for k,v in buckets.items()},
                 aggregate_occupancy_exact=True)
    save(ex/'summary.json',summary)
    lines=['# Validated T4096 online SPMCP accounting','',
           f'Cache: {cache:.4f} us; SPM: {spm:.4f} us; cache/SPM speedup: {cache/spm:.6f}x.',
           'These timings exactly reproduce the existing controls; instrumentation is observational.',
           f'All {g["count"]:,} copies accounted for; mean issue-to-install {summary["mean_copy_cycles"]:.3f} cycles.',
           'Each core’s admission-to-completion sum exactly matches the independently accumulated outstanding-request area.',
           '', '| Component | Share of summed copy latency |','|---|---:|']
    lines += [f'| {k} | {100*v/g["total_ticks"]:.2f}% |' for k,v in sorted(buckets.items(),key=lambda x:-x[1])]
    lines += ['', 'These are not FA runtime fractions or causal speedup predictions. The FIFO category needs head-of-queue evidence to separate downstream causes; it must not be labeled entirely as WB-credit stalls.',
              'Detailed breakdown: [SPM report](spm_t4096/BREAKDOWN.md). No full per-copy event trace was stored.','']
    (ex/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
