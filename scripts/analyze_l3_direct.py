"""Accepted-only transport diagnostics; never interpret stall sums as ROI shares."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks/cacheflex_flash_attention')]
from run_access_timing import stats
from build_access_timing import save

def analyze(out):
    state=json.loads((out/'status.json').read_text());rows=[]
    for r in state['completed']:
        path=Path(r['path'])
        assert json.loads((path/'accepted.json').read_text())==r
        s=stats(path)[0];total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
        n=total('.reductionAccepted')
        rows.append(dict(name=r['name'],case=r['case'],roi_us=r['roi_us'],commands=n,
            noc_flits=s.get('system.ruby.network.flits_injected::total'),
            direct_beats=total('.directBeats'),direct_bytes=r['direct_bytes'],
            mean_command_lifetime_cycles=total('.reductionLifetimeCycles')/n if n else None,
            source_full_retries=total('.reductionSourceFull'),
            dram_read_bytes=total('.dram.bytesRead::total'),dram_write_bytes=total('.dram.bytesWritten::total')))
    byname={r['name']:r for r in rows};effects=[]
    lines=['# Direct-link accepted FA effects','',f"Stage: {state['stage']}",'',
        '| T | NoC L3 us | Direct L3 us | Reduction vs NoC | vs private SPM | vs tuned cache |',
        '|---|---:|---:|---:|---:|---:|']
    for t in (1024,2048,4096):
        b=byname.get(f't{t}_noc');d=byname.get(f't{t}_direct')
        if not b or not d:continue
        e=dict(t=t,noc_us=b['roi_us'],direct_us=d['roi_us'],latency_reduction_pct=100*(1-d['roi_us']/b['roi_us']),speedup=b['roi_us']/d['roi_us'])
        values=[]
        for arm in ('private','cache'):
            x=byname.get(f't{t}_{arm}')
            if x:
                e[arm+'_us']=x['roi_us'];e['vs_'+arm+'_pct']=100*(1-d['roi_us']/x['roi_us']);values.append(f"{e['vs_'+arm+'_pct']:.3f}%")
            else:values.append('new-ELF repeat pending')
        lines.append(f"| {t} | {b['roi_us']:.4f} | {d['roi_us']:.4f} | {e['latency_reduction_pct']:.3f}% | {values[0]} | {values[1]} |")
        e['noc_flit_reduction_pct']=100*(1-d['noc_flits']/b['noc_flits'])
        e['mean_command_cycles']=[b['mean_command_lifetime_cycles'],d['mean_command_lifetime_cycles']]
        effects.append(e)
    lines+=['','## Transport diagnostics','',
        '| Case | NoC injected flits | Dedicated link beats | Mean command cycles | Source-full retries |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        if not r['name'].startswith(('t1024_','t2048_','t4096_')):continue
        mean='n/a' if r['mean_command_lifetime_cycles'] is None else f"{r['mean_command_lifetime_cycles']:.2f}"
        lines.append(f"| {r['name']} | {r['noc_flits']:.0f} | {r['direct_beats']:.0f} | {mean} | {r['source_full_retries']:.0f} |")
    lines+=['','Only accepted complete-ROI cases are included. Full oracle/exact bits, configuration, packet/copy counts and finite bounds passed.',
        'NoC flits exclude direct-link traffic; the dedicated beats are real serialized transfers, not eliminated work.',
        'Command lifetimes and source retries overlap computation and each other. They are not additive runtime fractions.',
        'NoC L3 and direct L3 use exactly the same kernel; cache/private use their independently best accepted bounded-search choices.',
        'Nominal: 128bit each direction, 2GHz, 2cy wire +1cy TX/RX endpoints, one assembly credit/direction, 2304B data buffering across8pairs.',
        'Slow sensitivity: 64bit and4cy propagation, unchanged engine/queues/SRAM. Explicit physical-model assumptions, not calibrated area/timing.',
        'Bank ALU/service and coherence/claim/output paths remain shared and bounded. No end-to-end Qwen model or energy claim.']
    save(out/'diagnostics.json',dict(rows=rows,effects=effects));(out/'EFFECTS.md').write_text('\n'.join(lines)+'\n')
    return state['stage']
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/cacheflex_l3_spm/direct_link_20260928/experiment_v2');ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    while True:
        stage=analyze(args.out)
        if not args.watch or stage in ('completed_validated','failed'):break
        time.sleep(15)
