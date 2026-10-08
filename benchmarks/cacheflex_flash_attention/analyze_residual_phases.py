"""Use verified debug ELFs, retaining per-core attribution and empty/idle bins."""
import json
from pathlib import Path
import re
import subprocess
import sys
import analyze_attribution as a
from analyze_residency import phase as cache_phase
from analyze_spm_residency import spm_phase

def analyze(run, guest, debug, variant, out):
    out.mkdir()
    for p,name in ((guest,'guest'),(debug,'debug')):
        subprocess.run(['aarch64-linux-gnu-objcopy','--dump-section',f'.text={out/name}.text',
                        str(p),str(out/(name+'.elf'))],check=True)
    assert (out/'guest.text').read_bytes()==(out/'debug.text').read_bytes()
    classifier=spm_phase if variant=='spm' else cache_phase
    a.phase=lambda frames,shift=0: classifier(frames)
    result=a.analyze(run,debug)
    lookup={int(p['pc'],16):p['phase'] for p in result['pcs']}
    cores={str(i):{} for i in range(8)}
    for line in (run/'stats.txt').read_text().split('End Simulation Statistics')[0].splitlines():
        w=line.split()
        if len(w)<2:continue
        m=re.fullmatch(r'system.cpu(\d+).commit.(pcHeadCycles|pcSleepHeadCycles)::(\d+)',w[0])
        if m:
            bins=cores[m[1]];phase=lookup[int(m[3])]
            bins[phase]=bins.get(phase,0)+int(w[1])
    result['per_core_attributed_cycles']=cores
    (out/'attribution.json').write_text(json.dumps(result,indent=2))
    print(variant,result['traffic']['roi_us'],
          {k:round(v['attributed_cycles']*.0004/8,3) for k,v in result['phases'].items()})
    return result

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    study=root/'results/cacheflex_flash_attention/causal_20260910_v3'
    syms=root/'results/cacheflex_flash_attention/residency_symbols_20260909'
    out=Path(sys.argv[1]);out.mkdir()
    for v in ('cache','spm'):
        analyze(study/f'factorial_{v}_t1024_m00_r0',study/v,syms/(v+'.debug'),v,out/v)
