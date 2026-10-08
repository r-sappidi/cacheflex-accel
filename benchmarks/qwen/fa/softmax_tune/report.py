"""FA-51 report from accepted runs only."""
import json
from pathlib import Path
ROOT = Path('/data2/rsappidi/cacheflex_mc/results/cacheflex_flash_attention/softmax_tune_20261005')
MEANING = {0: 'control', 1: 'four-row softmax', 2: 'skip unit alpha', 4: 'QK-epilogue max', 8: 'mask folded', 15: 'combined (1+2+4+8)'}


def main():
    s = json.loads((ROOT / 'status.json').read_text())
    rows = {r['name']: r for r in s['completed']}
    def get(t, spm, bc, r, f, cr): return rows.get(f"t{t}_{'spm' if spm else 'cache'}_bc{bc}_r{r}_f{f}_cr{cr}")
    L = ['# FA-51: per-flag attribution at T4096 and Bc/R retuning with flag 15', '',
         f"Stage {s['stage']}; accepted {len(rows)}; failed {len(s['failed'])}. All rows: full oracle, generalized work/copy",
         'counts, exact config, exact output bits per (T,Bc) versus FA-35 references or accepted peers.', '',
         '## T4096 per-flag attribution (Br16/Bc256/R256, 8 credits)', '', '| Flags | Meaning | Cache us | vs f0 | SPM us | vs f0 |', '|---:|---|---:|---:|---:|---:|']
    c0, s0 = get(4096, False, 256, 256, 0, 8), get(4096, True, 256, 256, 0, 8)
    for f, m in MEANING.items():
        c, p = get(4096, False, 256, 256, f, 8), get(4096, True, 256, 256, f, 8)
        fmt = lambda r, b: (f"{r['roi_us']:.1f}", f"{100 * (1 - r['roi_us'] / b['roi_us']):.2f}%" if b else '') if r else ('pending', '')
        cc, pp = fmt(c, c0), fmt(p, s0); L.append(f'| {f} | {m} | {cc[0]} | {cc[1]} | {pp[0]} | {pp[1]} |')
    L += ['', '## Retuning (T1024 coordinate descent, flag 15; Br16 fixed)', '']
    for key, d in sorted(s.get('descent', {}).items()):
        L.append(f"- {key}: path {d['path']}; evaluated us {d['evaluated']}")
    L += ['', '## T4096 confirmation', '', '| Backend/credits | T1024 winner | T4096 winner f15 us | T4096 runner-up f15 us | T4096 winner f0 us | Software gain |', '|---|---|---:|---:|---:|---:|']
    best = {}
    for key, d in sorted(s.get('confirm', {}).items()):
        spm = key.startswith('spm'); cr = int(key.split('cr')[1]); (bw, rw), (br_, rr) = d['winner_t1024'], d['runner_t1024']
        w, rn, z = get(4096, spm, bw, rw, 15, cr), get(4096, spm, br_, rr, 15, cr), get(4096, spm, bw, rw, 0, cr)
        cands = [x for x in (w, rn) if x];
        if cands: best[(spm, cr)] = min(cands, key=lambda x: x['roi_us'])
        gain = f"{100 * (1 - w['roi_us'] / z['roi_us']):.2f}%" if w and z else ''
        L.append(f"| {key} | Bc{bw}/R{rw} | {w['roi_us']:.1f} | {rn['roi_us'] if rn else 'n/a'} | {z['roi_us'] if z else 'n/a'} | {gain} |" if w else f'| {key} | Bc{bw}/R{rw} | pending | | | |')
    L += ['', '## Best tuned cache vs SPM at T4096 (flag 15)', '']
    for cr in (8, 16):
        c, p = best.get((False, cr)), best.get((True, cr))
        if c and p:
            L.append(f"- {cr} credits: cache {c['roi_us']:.1f} us (Bc{c['case']['bc']}/R{c['case']['r']}) vs SPM {p['roi_us']:.1f} us "
                     f"(Bc{p['case']['bc']}/R{p['case']['r']}): {c['roi_us'] / p['roi_us']:.4f}x")
    L += ['', 'Failed: ' + (json.dumps([dict(name=f['name'], error=f['error'][:200]) for f in s['failed']]) if s['failed'] else 'none'),
          '', 'Bounded Bc{128,256,512} x R{128,256,512} descent with Br16; 0.5% tie rule; not a global optimum.']
    (ROOT / 'REPORT.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
