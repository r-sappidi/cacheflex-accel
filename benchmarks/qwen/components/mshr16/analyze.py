"""AI-13 analysis of accepted runs only: speedups at 16 vs 8 L1D credits and SYN-02 energy.

Safe to run at any time; reads status/accepted markers, never launches simulations.
"""
import csv, json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
R = Path('/data2/rsappidi/cacheflex_mc')
sys.path[:0] = [str(H), str(R / 'scripts'), str(R / 'benchmarks/ai_workloads'), str(R / 'benchmarks/cacheflex_flash_attention')]
import plan as P
from run_access_timing import stats

# SYN-02: original paper RTL per-access energies (nJ) and idle powers (mW), unchanged.
E = dict(l1_read=0.3609, l1_write=0.4824, l2_read=1.1205, l2_write=1.4409, l3_read=5.121, l3_write=6.8205, spm_read=0.5328, spm_write=0.5526)
IDLE_L1, IDLE_L2_BL, IDLE_L2_CF, IDLE_L3 = 12.0, 26.0, 31.0, 51.0


def tot(ep, fmt):
    return sum(ep.get(fmt.format(i), 0.0) for i in range(8))


def energy(ep, us, spm):
    l1 = tot(ep, 'system.ruby.l0_cntrl{}.Dcache.m_demand_accesses')
    loads, stores = tot(ep, 'system.cpu{}.commitStats0.numLoadInsts'), tot(ep, 'system.cpu{}.commitStats0.numStoreInsts')
    spm_rd = tot(ep, 'system.cpu{}.spmLsq0.spmLoadResponses'); copies = tot(ep, 'system.ruby.l0_cntrl{}.sequencer.m_spmcpDataRequestsCompleted')
    sf = stores / max(1.0, loads - spm_rd + stores)
    parts = dict(
        l1=(l1 * (1 - sf) * E['l1_read'] + l1 * sf * E['l1_write']) / 1e6,
        l2=(tot(ep, 'system.ruby.l1_cntrl{}.cache.m_demand_accesses') * E['l2_read'] + tot(ep, 'system.ruby.l0_cntrl{}.writeCredits.allocations') * E['l2_write']) / 1e6,
        llc=(tot(ep, 'system.ruby.l2_cntrl{}.L2cache.m_demand_accesses') * E['l3_read'] + tot(ep, 'system.ruby.l1_cntrl{}.writeCredits.allocations') * E['l3_write']) / 1e6,
        spm=(spm_rd * E['spm_read'] + copies * (E['spm_write'] + E['l2_read'])) / 1e6,
        idle=(8 * (IDLE_L1 + (IDLE_L2_CF if spm else IDLE_L2_BL)) + IDLE_L3) * us * 1e-6,
        dram=sum(ep.get(f'system.mem_ctrls{c}.dram.rank{k}.totalEnergy', 0.0) for c in range(8) for k in range(2)) / 1e9)
    parts['total'] = sum(parts.values()); parts['onchip'] = parts['total'] - parts['dram']
    return parts


def old_rows():
    """Accepted AI-10/AI-11 target-shape rows (8 credits) for comparison."""
    out = []
    for wl, root in (('ffn', P.AI10 / 'experiment_v1/runs'), ('qkv', P.AI11 / 'experiment_v2/runs'), ('out', P.AI11 / 'experiment_v2/runs')):
        for acc in sorted(root.glob('*/accepted.json')):
            a = json.loads(acc.read_text()); c = a['case']
            if c.get('t') not in P.SIZES: continue
            if wl == 'ffn' and ((c.get('d'), c.get('f'), c.get('g')) != (896, 4864, 0)): continue
            if wl != 'ffn' and c.get('op') != wl: continue
            out.append(dict(source='AI-10' if wl == 'ffn' else 'AI-11', wl=wl, t=c['t'], arm=c['arm'], mp=c['mp'], kc=c['kc'], credits=8,
                            first_us=a['first_us'], repeat_us=a['repeat_us'], path=a['path'], name=a['name']))
    return out


def main():
    st = json.loads((P.OUT / 'status.json').read_text())
    rows = []
    for r in st['completed']:
        a = json.loads((Path(r['path']) / 'accepted.json').read_text()); c = a['case']
        if c['stage'] == 'gate' or c['stage'] == 'replay': continue
        rows.append(dict(source='AI-13', wl=c['wl'], t=c['t'], arm=c['arm'], mp=c['mp'], kc=c['kc'], credits=c['credits'],
                         first_us=a['first_us'], repeat_us=a['repeat_us'], path=a['path'], name=a['name']))
    rows += old_rows()
    for r in rows:
        blocks = stats(Path(r['path']))
        for i, key in ((0, 'first'), (1, 'repeat')):
            e = energy(blocks[i], r[f'{key}_us'], r['arm'] == 'spm')
            r[f'{key}_energy_mJ'] = e['total']; r[f'{key}_onchip_mJ'] = e['onchip']
            if key == 'repeat': r.update({f'repeat_{k}_mJ': v for k, v in e.items() if k not in ('total', 'onchip')})
    fields = list(rows[0])
    with (P.OUT / 'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    effects = []
    for wl in P.WORKLOADS:
        for t in P.SIZES:
            for cr in (8, 16):
                for metric in ('first', 'repeat'):
                    pick = {}
                    for arm in P.ARMS:
                        pool = [r for r in rows if (r['wl'], r['t'], r['arm'], r['credits']) == (wl, t, arm, cr)]
                        if pool: pick[arm] = min(pool, key=lambda r: r[f'{metric}_us'])
                    if len(pick) < 2: continue
                    c, s = pick['cache'], pick['spm']
                    effects.append(dict(wl=wl, t=t, credits=cr, metric=metric, cache=c['name'], spm=s['name'],
                        cache_kc=c['kc'], spm_kc=s['kc'], cache_us=c[f'{metric}_us'], spm_us=s[f'{metric}_us'],
                        speedup=c[f'{metric}_us'] / s[f'{metric}_us'], cache_mJ=c[f'{metric}_energy_mJ'], spm_mJ=s[f'{metric}_energy_mJ'],
                        energy_reduction=1 - s[f'{metric}_energy_mJ'] / c[f'{metric}_energy_mJ'],
                        n_cache=len([r for r in rows if (r['wl'], r['t'], r['arm'], r['credits']) == (wl, t, 'cache', cr)]),
                        n_spm=len([r for r in rows if (r['wl'], r['t'], r['arm'], r['credits']) == (wl, t, 'spm', cr)])))
    (P.OUT / 'effects.json').write_text(json.dumps(effects, indent=2) + '\n')
    lines = ['# AI-13: Qwen FFN/projections, cache vs SPM at 16 L1D miss credits', '',
             f"Coordinator stage: {st['stage']}; accepted {len(st['completed'])}; failed/skipped {len(st['failed'])}.", '',
             'Best accepted tiling per backend within each (workload, T, credits). 8-credit rows include AI-10/AI-11 accepted runs',
             'and AI-13 T1024 cache attribution runs; 8-credit candidate sets differ from 16-credit ones (n shown).',
             'Energy: SYN-02 paper RTL model on Ruby counters, per call; NoC/coherence/TLB/SPM-LSQ/core not modeled.', '',
             '| Workload | T | Credits | Metric | Cache (KC) us | SPM (KC) us | Speedup | Energy reduction | n cache/SPM |', '|---|---:|---:|---|---|---|---:|---:|---|']
    for e in effects:
        lines.append(f"| {e['wl']} | {e['t']} | {e['credits']} | {e['metric']} | {e['cache_us']:.1f} (KC{e['cache_kc']}) | {e['spm_us']:.1f} (KC{e['spm_kc']}) | "
                     f"{e['speedup']:.3f}x | {100 * e['energy_reduction']:.1f}% | {e['n_cache']}/{e['n_spm']} |")
    lines += ['', 'First call includes packing/staging; repeat call is the second invocation (warm diagnostic, not a converged median).',
              'Bounded MP8 KC axis {128,256,512,1024}; no global-optimum claim. Single simulator build (AI-10/AI-11 lifecycle build_v5), VL2048.']
    (P.OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
