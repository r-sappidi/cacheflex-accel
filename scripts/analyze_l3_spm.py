"""Report accepted L3-SPM gates separately from matched FA measurements."""
import json,os
from pathlib import Path
from build_access_timing import save
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_l3_spm/initial_20260928'
OUT=BASE/os.environ.get('CACHEFLEX_L3_EXPERIMENT','experiment_v7')
def main():
    state=json.loads((OUT/'status.json').read_text())
    rows=[]
    for p in sorted((OUT/'runs').glob('*/accepted.json')):rows.append(json.loads(p.read_text()))
    text=['# Shared-L3 SPM validation and pilot results','',f"Queue: {state['stage']}",'',
          'No reduction hardware. Existing LLC data SRAM is repurposed, not expanded.',
          'Matched new performance cases use the same conservative 64B/cycle/bank service model.',
          'Default-off equivalence, directed tests, dependent-load probes and FA are separate experiments.','',
          '| Case | ROI us | Independent check |','|---|---:|---|']
    for r in rows:text.append(f"| {r['name']} | {r['roi_us']:.4f} | {'full FA outputs' if 'arithmetic' in r else 'directed exact values'} |")
    lookup={r['name']:r for r in rows};ratios=[]
    for t in (1024,4096):
        names=[f't{t}_{n}' for n in ('cache','private_spm','l3_output')]
        if all(n in lookup for n in names):
            a,b,c=[lookup[n]['roi_us'] for n in names]
            ratios.append(dict(t=t,private_over_cache=a/b,l3_over_cache=a/c,l3_over_private=b/c))
            text+=['',f'T{t}: cache/private-SPM {a/b:.4f}x; cache/L3-output {a/c:.4f}x; private-SPM/L3-output {b/c:.4f}x.']
    a,b=lookup.get('latency_m2_c1'),lookup.get('latency_l3_slow')
    if a and b:
        text+=['',f"Dependent-load response-latency sensitivity: {a['roi_us']:.4f} → {b['roi_us']:.4f} us."]
        assert b['roi_us']>a['roi_us'],'Expected increased dependent latency was not observed'
    text+=['','These are functionality/performance-sanity pilots, not independently tuned L3 FA or paper-ready speedups.',
           'Cross-core accesses require explicit producer/consumer synchronization; no atomics or OS isolation is claimed.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n');save(OUT/'analysis.json',dict(cases=rows,ratios=ratios))
    print(OUT/'REPORT.md')
if __name__=='__main__':main()
