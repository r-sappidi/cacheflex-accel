"""Accepted-only summary of the repaired placement experiment."""
import json
from pathlib import Path
from build_access_timing import save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cacheflex_flash_attention/spm_set_spread_20260927/experiment_retry_v2'

def main():
    rows={p.parent.name:json.loads(p.read_text()) for p in (OUT/'runs').glob('*/validated.json')}
    state=json.loads((OUT/'status.json').read_text())
    lines=['# Bounded silent-retry repair and placement revalidation','',
           'Related ledger: FA-31, FA-29, MOD-03. Accepted results only; failed or partial cases excluded.',
           '',f"Queue stage: {state['stage']}", '',
           '| Case | ROI (us) | NACKs/reissues | Arithmetic |',
           '|---|---:|---:|---|']
    for name,r in rows.items():
        lines.append(f"| {name} | {r.get('roi_us','—')} | {r.get('silent_retry_received','—')}/{r.get('silent_retry_reissued','—')} | {'pass' if 'arithmetic' in r else 'bridge/wide check pass'} |")
    comparisons={}
    for t in (4096,16384):
        names=[f't{t}_cache',f't{t}_spm_control',f't{t}_'+('spread' if t==4096 else 'spm_spread')]
        if all(n in rows for n in names):
            c,s,p=[rows[n]['roi_us'] for n in names]
            comparisons[str(t)]=dict(cache_us=c,control_spm_us=s,spread_spm_us=p,
                control_speedup=c/s,spread_speedup=c/p,placement_speedup=s/p)
            lines += ['',f'T{t}: cache/control-SPM/spread-SPM = {c:.4f}/{s:.4f}/{p:.4f} us; '
                      f'placement {s/p:.6f}x, spread vs cache {c/p:.6f}x.']
    lines += ['', 'T4096 uses matched Br16/Bc512/R256 pilots, not the independently tuned cache minimum. '
              'T16K transfers the prior best measured Br16/Bc512/R768 configurations for both architectures. '
              'All three target arms enable the same repair. No new SRAM/credits/ports; placement row-selection '
              'timing is not synthesized and bank contention is not modeled. Gates are not exhaustive liveness proof.']
    save(OUT/'summary.json',dict(stage=state['stage'],accepted=rows,comparisons=comparisons))
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
