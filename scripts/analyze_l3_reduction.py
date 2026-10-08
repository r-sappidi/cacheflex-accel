"""Summarize accepted matched near-L3 FA arms, never partial/crashed timings."""
import argparse, json, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def summarize(out):
    state=json.loads((out/'status.json').read_text())
    rows={}
    for file in (out/'runs').glob('*/accepted.json'):
        r=json.loads(file.read_text()); rows[r['name']]=r
    lines=['# Matched near-L3 reduction comparison','',f"Queue state: {state['stage']}",'',
        'Only accepted full-output/config/resource-checked runs are used.',
        'Speedup is baseline ROI / candidate ROI; greater than 1 is faster.',
        'Same-platform pilot configurations, not a claim of exhaustive best-to-best tuning.','']
    for t in (1024,4096):
        arms={n:rows.get(f't{t}_{n}') for n in
              ('cache','private_spm','l3_output','l3_async','l3_sync')}
        lines += [f'## T{t}','',
            '| Arm | ROI (us) | vs cache | vs private SPM | vs L3 storage-only |',
            '|---|---:|---:|---:|---:|']
        for n,r in arms.items():
            if r is None:
                lines.append(f'| {n} | not accepted | — | — | — |');continue
            ratios=[f"{arms[b]['roi_us']/r['roi_us']:.5f}x" if arms[b] else '—'
                    for b in ('cache','private_spm','l3_output')]
            lines.append(f"| {n} | {r['roi_us']:.4f} | {' | '.join(ratios)} |")
        lines.append('')
    lines += ['Finite extra buffers and arithmetic are modeled; no new accumulator SRAM.',
        'Read/write bank bandwidth and network payload costs remain charged.',
        'Timing parameters are explicit assumptions, not calibrated RTL.',
        'Request lifetimes and summed stalls are not additive removable runtime fractions.']
    text='\n'.join(lines)+'\n'
    for t in (1024,):
        # The new shared ELF can perturb even disabled code layout. Do not
        # hide a stronger, accepted same-platform original-guest control.
        for n in ('cache','private_spm'):
            candidates=[rows[k]['roi_us'] for k in (f't{t}_{n}',f'disabled_{n}') if k in rows]
            if candidates:
                text+=f'\nStrongest accepted T{t} {n} control (original/new ELF): {min(candidates):.4f} us.\n'
                for variant in ('l3_async','l3_sync'):
                    if f't{t}_{variant}' in rows:
                        text+=f"- {variant} speedup vs this control: {min(candidates)/rows[f't{t}_{variant}']['roi_us']:.5f}x.\n"
    (out/'COMPARISON.md').write_text(text);print(text)
    return state['stage']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',default='experiment_v3')
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    out=ROOT/'results/cacheflex_l3_spm/reduction_20260928'/a.experiment
    while True:
        stage=summarize(out)
        if not a.watch or stage in ('failed','completed_requires_analysis'):break
        time.sleep(30)
if __name__=='__main__':main()
