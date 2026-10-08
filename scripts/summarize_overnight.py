"""Read-only rolling comparison of accepted FA data; never launches simulations."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_fa_overnight_diagnosis import analyze, DEBUG, BUILDS, SWEEP, save

def compare(before,after):
    phases=[]
    for name in before['phases'].keys() | after['phases'].keys():
        x=before['phases'].get(name,{});y=after['phases'].get(name,{})
        phases.append(dict(phase=name,delta_mean_core_us=(y.get('attributed_cycles',0)-x.get('attributed_cycles',0))/20000))
    old={p['pc']:p for p in before['pcs']};pcs=[]
    for p in after['pcs']:
        q=old.get(p['pc'],{})
        pcs.append(dict(pc=p['pc'],instruction=p['instruction'],phase=p['phase'],
          delta_mean_core_us=(p['attributed_cycles']-q.get('attributed_cycles',0))/20000,
          delta_no_retire_mean_core_us=(p.get('pcNoRetireCycles',0)-q.get('pcNoRetireCycles',0))/20000))
    routes=[]
    for name in before['routes'].keys() | after['routes'].keys():
        x=before['routes'].get(name,{});y=after['routes'].get(name,{})
        routes.append(dict(route=name,delta_packets=y.get('packets',0)-x.get('packets',0),
          delta_queue_ticks=y.get('queue_ticks',0)-x.get('queue_ticks',0)))
    return dict(before_us=before['traffic']['roi_us'],after_us=after['traffic']['roi_us'],
      phases=sorted(phases,key=lambda p:-abs(p['delta_mean_core_us'])),
      pcs=sorted(pcs,key=lambda p:-abs(p['delta_mean_core_us'])),
      routes=sorted(routes,key=lambda p:-abs(p['delta_queue_ticks'])),
      traffic_before=before['traffic'],traffic_after=after['traffic'])

def main():
    out=Path(sys.argv[1]); derived=out/'rolling_analysis';derived.mkdir(exist_ok=True)
    seen={}
    while True:
        # Analyze newly completed sequencer runs without touching their files.
        for accepted in sorted((SWEEP/'runs').glob('*/accepted.json')):
            run=accepted.parent;dest=derived/(run.name+'.json')
            if dest.exists():continue
            arch=run.name.split('_')[0]
            debug=DEBUG if arch=='spm' else BUILDS/'fce06cc784b2725b/guest.debug'
            # Reuse initial analysis if available; all PCs were byte-verified.
            initial=out/'control_analysis'/(run.name+'.json')
            if initial.exists(): result=json.loads(initial.read_text())
            else:
                from analyze_fa_retention_pcs import a, phase, instruction_bytes
                a.phase=phase;result=a.analyze(run,debug)
                cmd=json.loads((run/'command.json').read_text())
                gd=Path(cmd[cmd.index('--cmd')+1]).read_bytes();dd=debug.read_bytes()
                for pc in result['pcs']:
                    addr=int(pc['pc'],16);assert instruction_bytes(gd,addr)==instruction_bytes(dd,addr)
            save(dest,result)
        for t in (1024,16384):
            for arch in ('cache','spm'):
                base=derived/f'{arch}_t{t}_seq16.json'
                if not base.exists():continue
                for n in (32,64,128):
                    other=derived/f'{arch}_t{t}_seq{n}.json'
                    if not other.exists():continue
                    name=f'{arch}_t{t}_16_to_{n}'
                    if name not in seen:
                        seen[name]=compare(json.loads(base.read_text()),json.loads(other.read_text()))
        save(derived/'sequencer_deltas.json',seen)
        lines=['# Accepted sequencer sensitivity: mechanism accounting','',
          'Phase/PC figures are mean-core attribution, not independently removable causal delays.',
          'Route queue time sums overlap across packets and cannot be converted into lost runtime.',
          'The intervention results and correctness status are in ../REPORT.md and ../status.json.','']
        for name,r in seen.items():
            lines += [f'## {name}','',f"Runtime: {r['before_us']:.4f} → {r['after_us']:.4f} us.",'',
              '| Phase | Change in mean-core us |','|---|---:|']
            for p in r['phases'][:6]:lines.append(f"| {p['phase']} | {p['delta_mean_core_us']:+.4f} |")
            lines.append('')
        tmp=derived/'REPORT.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(derived/'REPORT.md')
        states=[json.loads((p/'status.json').read_text())['stage'] for p in (SWEEP,out)]
        save(derived/'status.json',dict(updated=time.time(),observed_states=states,comparisons=len(seen)))
        if all(s in ('complete','failed') for s in states):return
        time.sleep(30)

if __name__=='__main__':main()
