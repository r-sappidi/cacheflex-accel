"""Accepted-only matched lifecycle deltas and traffic diagnostics."""
import argparse,json,time
from pathlib import Path
from run_access_timing import stats
from build_access_timing import save
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/cacheflex_l3_spm/lifecycle_20260928/experiment_v2'
def analyze(root):
    accepted={}
    for p in (root/'runs').glob('*/accepted.json'):
        row=json.loads(p.read_text());s=stats(p.parent)[0]
        total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
        n=row['commands']
        row['diagnostics']=dict(flits=s.get('system.ruby.network.flits_injected::total'),
            instructions=s.get('simInsts'),source_full=total('.reductionSourceFull'),
            mean_reduction_core_cycles=total('.reductionLifetimeCycles')/n if n else None,
            dram_read_bytes=total('.dram.bytesRead::total'),
            dram_write_bytes=total('.dram.bytesWritten::total'),
            writeback_count=s.get('system.ruby.RequestType.SPMWB_store.latency_hist_seqr::samples',0),
            writeback_mean_ruby_cycles=s.get('system.ruby.RequestType.SPMWB_store.latency_hist_seqr::mean'))
        accepted[row['name']]=row
    effects=[]
    for t in (1024,4096):
        for a,b,label in [('base','init','Initialize with first update'),
                          ('base','final','Normalize/writeback/release'),
                          ('init','both','Finalize after initialization'),
                          ('final','both','Initialize after finalization'),
                          ('base','both','Combined vs reduction'),
                          ('cache','both','Combined vs cache'),
                          ('private_spm','both','Combined vs private SPM')]:
            x=accepted.get(f't{t}_{a}');y=accepted.get(f't{t}_{b}')
            if x and y:
                effects.append(dict(t=t,comparison=label,control_us=x['roi_us'],candidate_us=y['roi_us'],
                    saved_us=x['roi_us']-y['roi_us'],latency_reduction_pct=100*(1-y['roi_us']/x['roi_us']),
                    speedup=x['roi_us']/y['roi_us']))
    save(root/'effects.json',dict(accepted=accepted,effects=effects))
    lines=['# Accepted lifecycle effects','',
        '| T | Comparison | Control us | Candidate us | Latency reduction | Speedup |',
        '|---|---|---:|---:|---:|---:|']
    for r in effects:
        lines.append(f"| {r['t']} | {r['comparison']} | {r['control_us']:.4f} | {r['candidate_us']:.4f} | {r['latency_reduction_pct']:.3f}% | {r['speedup']:.6f}x |")
    lines+=['','Matched interventions, not additive runtime fractions. Same-ELF controls; full numerical checks.',
        'Finalization includes the existing translated write-through/DRAM-ack policy.',
        'Writeback/reduction lifetime averages overlap computation and must not be summed into ROI.',
        'T1024 fixed pipelined FP16; T4096 fixed serialized FP16 (previous per-size winners).']
    (root/'EFFECTS.md').write_text('\n'.join(lines)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        if (a.root/'status.json').exists():
            analyze(a.root)
            stage=json.loads((a.root/'status.json').read_text())['stage']
            if stage in ('completed_requires_analysis','failed'):break
        if not a.watch:break
        time.sleep(20)
if __name__=='__main__':main()
