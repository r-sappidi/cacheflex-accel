"""Summarize accepted end-to-end measurements without causal-cycle claims."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/cacheflex_flash_attention/micro_resources_20260921'

def main():
    lines=['# MICRO-derived Ruby measurements','',
           'Only accepted full-reference cases are listed. One timing per point;',
           'guard repeats test determinism, not independent statistical replicates.',
           'This is a resource translation, not cycle-identical Classic reproduction.','',
           '| Configuration | T | Cache µs | SPM µs | Cache/SPM |',
           '|---|---:|---:|---:|---:|']
    rows=[]
    for label,folder,kind in [
        ('Original shared16',ROOT/'results/cacheflex_flash_attention/async_staging_20260921/experiment','original'),
        ('Prior finite model',ROOT/'results/cacheflex_flash_attention/hardware_bounds_20260921/experiment_v5','bounded'),
        ('MICRO resources,64KiB L2',BASE/'experiment_v6','bounded'),
        ('MICRO resources,512KiB L2/no retention',BASE/'experiment_paper_geometry_v2','bounded')]:
        for t in (512,4096):
            pair={}
            for arch in ('cache','spm'):
                name=f'{arch}_t{t}' if kind=='original' else f'{arch}_{kind}_t{t}'
                p=folder/'runs'/name/'accepted.json'
                if p.exists(): pair[arch]=json.loads(p.read_text())
            if len(pair)!=2: continue
            c,s=(pair[a]['counters']['roi_us'] for a in ('cache','spm'))
            lines.append(f'| {label} | {t} | {c:.4f} | {s:.4f} | {c/s:.6f}× |')
            rows.append(dict(label=label,T=t,cache_us=c,spm_us=s,speedup=c/s,
                             sources={a:v['path'] for a,v in pair.items()}))
    lines += ['', 'The new profile changes cache-side credit accounting, SPM ingress and',
              'SPM load issue ports together. These are not separately attributed savings.',
              'The512KiB comparison also disables LLC retention and must not be described',
              'as an isolated capacity-only or resource-only intervention.']
    (BASE/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n')
    (BASE/'measurements.json').write_text(json.dumps(rows,indent=2))
    print('\n'.join(lines))

if __name__=='__main__': main()
