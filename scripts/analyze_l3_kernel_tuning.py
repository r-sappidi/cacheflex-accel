"""Accepted-only effects and final ledger update for the ordered kernel study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data2/rsappidi/cacheflex_mc')
OUT=ROOT/'results/cacheflex_l3_spm/kernel_tuning_20260928/experiment_v1'


def analyze():
    state=json.loads((OUT/'status.json').read_text())
    rows=state['completed'];effects=[]
    def add(label,control,candidate):
        a,b=control['roi_us'],candidate['roi_us']
        effects.append(dict(comparison=label,control=control['name'],candidate=candidate['name'],
                            control_us=a,candidate_us=b,latency_reduction_pct=100*(1-b/a),speedup=a/b))
    for t in (1024,4096):
        for arm in ('cache','private','l3'):
            selected=[r for r in rows if r['case']['t']==t and r['case']['arm']==arm and r['case']['depth']==4
                      and r['case']['bc']==(128 if t==1024 else 512)
                      and r['case']['br']==(32 if t==1024 else 16)
                      and r['case']['r']==(192 if t==1024 else 256)]
            byk={r['case']['k']:r for r in selected}
            if 0 in byk and 1 in byk:add(f'T{t} {arm}: direct epilogue',byk[0],byk[1])
        best={}
        for arm in ('cache','private','l3'):
            selected=[r for r in rows if r['case']['t']==t and r['case']['arm']==arm and r['case']['depth']==4]
            if selected:best[arm]=min(selected,key=lambda r:r['roi_us'])
        if 'l3' in best:
            for arm in ('cache','private'):
                if arm in best:add(f'T{t} current best L3 vs {arm}',best[arm],best['l3'])
    doc=['# Accepted L3 kernel-tuning effects','',f"Stage: {state['stage']}",'',
         '| Comparison | Control us | Candidate us | Latency reduction | Speedup |',
         '|---|---:|---:|---:|---:|']
    for e in effects:
        doc.append(f"| {e['comparison']} | {e['control_us']:.4f} | {e['candidate_us']:.4f} | {e['latency_reduction_pct']:.3f}% | {e['speedup']:.6f}x |")
    doc += ['', 'Negative reduction means a regression. Each row is a measured end-to-end comparison; effects are not additive.',
            'The direct epilogue also changes independent-row submission order to paired halves; phase 3 tests sequential order.',
            'Private and L3 get the same bounded tile/group search budgets; cache epilogue controls are fixed-tile only.',
            'Retain historical private-SPM placement results as additional references; this search does not prove a global optimum.',
            'Full arithmetic, exact same-Bc output, configuration, command/copy and finite-resource checks required.',
            'A failed queue does not invalidate earlier accepted cases, nor validate its incomplete cases.']
    (OUT/'EFFECTS.md').write_text('\n'.join(doc)+'\n')
    (OUT/'effects.json').write_text(json.dumps(dict(stage=state['stage'],effects=effects),indent=2)+'\n')
    if state['stage']=='completed_validated' and not (OUT/'ledger_recorded.json').exists():
        ledger=ROOT/'docs/experiments/LEDGER.md';text=ledger.read_text()
        update=['','**L3-05 completion update:** all '+str(len(rows))+' cases accepted with full checks.']
        for e in effects:
            if e['comparison'].startswith('T4096'):
                update.append(f"{e['comparison']}: {e['control_us']:.4f}->{e['candidate_us']:.4f}us; {e['latency_reduction_pct']:.3f}% lower latency ({e['speedup']:.6f}x).")
        update += ['Bounded software search; original hardware resources unchanged. Read the effects report for negative and positive comparisons.','']
        anchor='## Deliberation template and maintenance'
        assert text.count(anchor)==1
        ledger.write_text(text.replace(anchor,'\n'.join(update)+'\n'+anchor))
        (OUT/'ledger_recorded.json').write_text(json.dumps(dict(time=time.time(),cases=len(rows)))+'\n')
        with (OUT/'inventory_refresh.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/index_experiment_history.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return state['stage']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        try:stage=analyze()
        except (FileNotFoundError,json.JSONDecodeError):stage='waiting'
        if not a.watch or stage in ('failed','completed_validated'):break
        time.sleep(15)
