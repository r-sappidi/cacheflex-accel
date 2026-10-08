"""Read-only summary of accepted targets; never promote gate timings."""
import json
from pathlib import Path
import sys

def main(path):
    state=json.loads((path/'status.json').read_text())
    rows=[r for r in state['completed'] if r['case']['stage']=='target']
    print('Queue:',state['stage'],'; accepted targets:',len(rows))
    print('Ratios below concern this new driver, NOT established best-to-best.')
    for t,n in sorted({(r['case']['t'],r['case']['n']) for r in rows}):
        relevant=[r for r in rows if (r['case']['t'],r['case']['n'])==(t,n)]
        best={}
        for kv in (False,True):
            for spm in (False,True):
                candidates=[r for r in relevant if (r['case']['kv'],r['case']['spm'])==(kv,spm)]
                if candidates:
                    r=min(candidates,key=lambda x:x['roi_us']);best[kv,spm]=r
                    print(f"T{t} n{n} {'KV' if kv else 'Q'} {'SPM' if spm else 'cache'}: "
                          f"{r['roi_us']:.4f} us, G={r['case']['group']}, copies={r['copies']}")
        for kv in (False,True):
            if (kv,False) in best and (kv,True) in best:
                ratio=best[kv,False]['roi_us']/best[kv,True]['roi_us']
                print(f'  Same-mapping cache/SPM ratio ({kv=}): {ratio:.6f}x')
        caches=[r for (kv,spm),r in best.items() if not spm]
        if caches and (True,True) in best:
            cache=min(caches,key=lambda r:r['roi_us'])
            ratio=cache['roi_us']/best[True,True]['roi_us']
            print(f'  Best screened cache / persistent-V SPM: {ratio:.6f}x')

if __name__=='__main__':main(Path(sys.argv[1]))
