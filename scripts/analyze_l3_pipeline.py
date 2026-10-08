"""Accepted-only incremental effects; queue waits/stats are not runtime shares."""
import argparse,json,time
from pathlib import Path
from run_access_timing import stats
from build_access_timing import save
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/cacheflex_l3_spm/pipeline_20260928/experiment_v2'

def analyze(root):
    accepted={}
    for p in (root/'runs').glob('*/accepted.json'):
        row=json.loads(p.read_text());s=stats(p.parent)[0]
        total=lambda suffix:sum(v for k,v in s.items() if k.endswith(suffix))
        row['diagnostics']=dict(flits=s.get('system.ruby.network.flits_injected::total'),
            instructions=s.get('simInsts'),source_full=total('.reductionSourceFull'),
            mean_command_cycles=(total('.reductionLifetimeCycles')/row['commands'] if row['commands'] else None))
        accepted[row['name']]=row
    effects=[]
    for t in (1024,4096):
        def compare(a,b,label):
            x=accepted.get(f't{t}_{a}');y=accepted.get(f't{t}_{b}')
            if x and y:
                effects.append(dict(t=t,comparison=label,control_us=x['roi_us'],candidate_us=y['roi_us'],
                    saved_us=x['roi_us']-y['roi_us'],latency_reduction_pct=100*(1-y['roi_us']/x['roi_us']),
                    speedup=x['roi_us']/y['roi_us']))
        compare('base','pipeline','Pipelining alone')
        compare('base','half','FP16 transport alone')
        compare('pipeline','both','FP16 added after pipelining')
        compare('half','both','Pipelining added after FP16')
        compare('base','both','Combined vs serialized reduction')
        compare('cache','both','Combined vs cache')
        compare('private_spm','both','Combined vs private SPM')
    payload=dict(accepted=accepted,effects=effects)
    save(root/'effects.json',payload)
    lines=['# Accepted incremental end-to-end effects','',
        '| T | Comparison | Control us | Candidate us | Latency reduction | Speedup |',
        '|---|---|---:|---:|---:|---:|']
    for r in effects:
        lines.append(f"| {r['t']} | {r['comparison']} | {r['control_us']:.4f} | {r['candidate_us']:.4f} | {r['latency_reduction_pct']:.3f}% | {r['speedup']:.6f}x |")
    lines+=['','Each row is a matched counterfactual difference, not an additive fraction of total runtime.',
        'Do not sum the isolated gains: overlapping bottlenecks can mask or amplify a combined change.',
        'Typed transport also removes CPU widening and adds bank conversion; this is a complete mechanism comparison, not a network-only latency claim.',
        'Use REPORT.md for old-ELF controls and preserve stronger historical private baselines.']
    (root/'EFFECTS.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        analyze(a.root)
        stage=json.loads((a.root/'status.json').read_text())['stage']
        if not a.watch or stage in ('completed_requires_analysis','failed'):break
        time.sleep(20)
if __name__=='__main__':main()
