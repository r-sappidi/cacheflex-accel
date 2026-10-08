"""Merged AI-13 + AI-13b + AI-10/11 analysis (accepted runs only; no simulation).

Best tiling per backend within each (workload, T, credits), first and repeat call,
with the number of tilings tested so tuning coverage is visible. Writes
results/ai_workloads/qwen_mshr16_neighbors_20261005/MERGED_REPORT.md and merged.json.
"""
import json, sys
from pathlib import Path
R = Path('/data2/rsappidi/cacheflex_mc')
sys.path.insert(0, str(R / 'benchmarks/qwen/components/mshr16'))
import plan as P
import analyze as A                    # AI-13 analysis helpers: old_rows(), energy(), stats
OUT = R / 'results/ai_workloads/qwen_mshr16_neighbors_20261005'


def accepted(status_path, source):
    if not Path(status_path).exists(): return []
    rows = []
    for r in json.loads(Path(status_path).read_text())['completed']:
        c = r['case']
        if c['stage'] in ('gate', 'replay'): continue
        a = json.loads((Path(r['path']) / 'accepted.json').read_text())
        rows.append(dict(source=source, wl=c['wl'], t=c['t'], arm=c['arm'], mp=c['mp'], kc=c['kc'], credits=c['credits'],
                         first_us=a['first_us'], repeat_us=a['repeat_us'], path=a['path'], name=a['name']))
    return rows


def main():
    rows = accepted(P.OUT / 'status.json', 'AI-13') + accepted(OUT / 'status.json', 'AI-13b') + A.old_rows()
    for r in rows:
        blocks = A.stats(Path(r['path']))
        r['repeat_energy_mJ'] = A.energy(blocks[1], r['repeat_us'], r['arm'] == 'spm')['total']
    table = []
    for wl in P.WORKLOADS:
        for t in P.SIZES:
            for cr in (8, 16):
                for metric in ('first', 'repeat'):
                    pick = {}
                    for arm in P.ARMS:
                        pool = [r for r in rows if (r['wl'], r['t'], r['arm'], r['credits']) == (wl, t, arm, cr) and r['mp'] == 8]
                        if pool: pick[arm] = (min(pool, key=lambda r: r[f'{metric}_us']), sorted({r['kc'] for r in pool}))
                    if len(pick) < 2: continue
                    (c, ck), (s, sk) = pick['cache'], pick['spm']
                    table.append(dict(wl=wl, t=t, credits=cr, metric=metric, cache_us=c[f'{metric}_us'], cache_kc=c['kc'], cache_tested=ck,
                                      spm_us=s[f'{metric}_us'], spm_kc=s['kc'], spm_tested=sk, speedup=c[f'{metric}_us'] / s[f'{metric}_us'],
                                      cache_converged=_converged(c['kc'], ck), spm_converged=_converged(s['kc'], sk)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'merged.json').write_text(json.dumps(dict(rows=rows, table=table), indent=1) + '\n')
    lines = ['# AI-13 merged: best tested MP8 tiling per backend, 8 vs 16 L1D credits', '',
             'Converged = best KC has both neighbors in {128,256,512,1024} tested (or is at the set edge).', '',
             '| Workload | T | Credits | Call | Cache us (KC; tested) | SPM us (KC; tested) | Speedup | Converged cache/SPM |',
             '|---|---:|---:|---|---|---|---:|---|']
    for e in table:
        lines.append(f"| {e['wl']} | {e['t']} | {e['credits']} | {e['metric']} | {e['cache_us']:.1f} (KC{e['cache_kc']}; {e['cache_tested']}) | "
                     f"{e['spm_us']:.1f} (KC{e['spm_kc']}; {e['spm_tested']}) | {e['speedup']:.3f}x | {e['cache_converged']}/{e['spm_converged']} |")
    (OUT / 'MERGED_REPORT.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


def _converged(kc, tested):
    i = P.KCS.index(kc)
    return all(P.KCS[j] in tested for j in (i - 1, i + 1) if 0 <= j < len(P.KCS))


if __name__ == '__main__':
    main()
