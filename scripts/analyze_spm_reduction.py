"""Read accepted cases only; report end-to-end ratios, never sum stall counters."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/near_spm_reduction_20260927'
OLD=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/experiment_retry_v2/runs'
def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',default='experiment');a=p.parse_args()
    out=BASE/a.experiment
    rows=[json.loads(f.read_text()) for f in (out/'runs').glob('*/accepted.json')]
    text=['# FA accumulation: end-to-end comparison','',
          'Only accepted full-output/config/count checks are included. T1024 is a correctness gate, not the target workload.',
          'Same current model; original frozen kernels remain eligible controls. Separate timings are not additive latency fractions.','']
    for t in (1024,4096,16384):
        cases=[r for r in rows if r['job'].get('t')==t]
        if not cases:continue
        controls=[]
        for arch,label in [('cache','cache'),('spm','spm_control')]:
            name=f'repaired_{arch}' if t==1024 else f't{t}_{label}'
            src=OLD/name/'validated.json'
            if src.exists():
                r=json.loads(src.read_text());controls.append((arch,'frozen_'+arch,r['roi_us']))
        if t==16384:
            r=json.loads((OLD/'t16384_spm_spread/validated.json').read_text())
            controls.append(('spm','frozen_spm_spread',r['roi_us']))
        for r in cases:
            j=r['job']
            if j['mode']<3:controls.append(('spm' if j['spm'] else 'cache',r['name'],r['roi_us']))
        cache=min((x for x in controls if x[0]=='cache'),key=lambda x:x[2])
        spm=min((x for x in controls if x[0]=='spm'),key=lambda x:x[2])
        text += [f'## T{t}','',f'Best validated cache reference: {cache[1]}, {cache[2]:.4f} us.',
                 f'Best validated non-reduction SPM reference: {spm[1]}, {spm[2]:.4f} us.','',
                 '| Variant | ROI us | Cache / variant | SPM / variant |',
                 '|---|---:|---:|---:|']
        for r in sorted(cases,key=lambda r:r['name']):
            v=r['roi_us'];text.append(f"| {r['name']} | {v:.4f} | {cache[2]/v:.6f}x | {spm[2]/v:.6f}x |")
        if t==4096:text+=['','The frozen T4096 cache here is a matched R256 pilot, not the independent R192 search minimum. A gain against only this pilot is not a best-to-best claim.']
    text+=['','Limits: conservative existing timed SPM read/write backend, no calibrated bank-level arbitration; extra bounded command buffers and ALU are additional hardware.']
    report='\n'.join(text)+'\n';(out/'COMPARISON.md').write_text(report);print(report)
if __name__=='__main__':main()
