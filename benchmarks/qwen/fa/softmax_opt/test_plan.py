"""Host-only FA-50 tests; no simulation."""
import json, os, shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent; R = H.parents[3]
os.environ.setdefault('FA50_LANE', '0')
sys.path.insert(0, str(H))
import plan as P
import run as RUN
FA39 = R / 'results/cacheflex_flash_attention/bottleneck_20261004/experiment_v1/runs'


def test_plan():
    rows = P.cases(); names = [c['name'] for c in rows]
    assert len(names) == len(set(names)) == 2 * len(P.FLAGS) + 2 * len(P.TIMING)
    lanes = P.lanes(); assert sorted(c['name'] for l in lanes for c in l) == sorted(names)
    for l in lanes:  # within a lane, gates precede timing cases
        stages = [c['stage'] for c in l]; assert stages == sorted(stages, key=lambda s: s != 'gate')
    print('plan ok:', len(rows), 'cases in 4 lanes')


def test_command_matches_fa39():
    tmp = Path(tempfile.mkdtemp())
    try:
        for spm, old in ((False, 'gate_cache_control'), (True, 'gate_spm_control')):
            c = dict(t=256, spm=spm, flags=0, stage='gate'); c['name'] = P.name(c)
            new = RUN.command(c, tmp); ref = json.loads((FA39 / old / 'command.json').read_text())
            diff = [(a, b) for a, b in zip(new, ref) if a != b]
            assert len(new) == len(ref)
            allowed = {'--outdir', '--cmd-guest', '--options', 'system.cpu[0].workload[0].cmd'}
            for a, b in diff:
                if a.startswith('--outdir='): continue
                if a.endswith('guest_v1/guest.export-safe') and b.endswith('guest_v1/guest.export-safe'): continue
                if a == b + ' 0': continue                                  # --options gains flag word
                if a.startswith('system.cpu[0].workload[0].cmd=') and a.replace(", '0']", "]") == b: continue
                raise SystemExit(f'unexpected command difference: {a!r} vs {b!r}')
    finally:
        shutil.rmtree(tmp)
    print('commands differ from FA-39 only by outdir, guest path and appended flag word')


def test_config_negative():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = dict(t=256, spm=True, flags=15, stage='gate'); c['name'] = P.name(c)
        cfg = json.loads((RUN.PARENT / 'config.json').read_text().replace(str(RUN.PARENT), str(tmp)))
        s = cfg['system']; s['cpu'][0]['workload'][0]['executable'] = str(RUN.G / 'guest.export-safe')
        s['cpu'][0]['workload'][0]['cmd'] = ['kv-fa', *map(str, P.argv(c))]
        for i in range(8): s['ruby'][f'l1_cntrl{i}']['cache']['num_spm_ways'] = 6
        (tmp / 'config.json').write_text(json.dumps(cfg)); cmd = RUN.command(c, tmp)
        RUN.config_check(tmp, c, cmd)
        s['ruby']['l0_cntrl0']['dataMissCredits']['capacity'] = 16; (tmp / 'config.json').write_text(json.dumps(cfg))
        try: RUN.config_check(tmp, c, cmd)
        except AssertionError: pass
        else: raise SystemExit('config check accepted a hardware change')
    finally:
        shutil.rmtree(tmp)
    print('config check accepts the planned case and rejects a hardware change')


def test_guest():
    g = RUN.G
    for p, h in json.loads((g / 'frozen.json').read_text()).items(): assert RUN.sha(p) == h, p
    drv = (g / 'driver.cpp').read_text()
    assert drv.count('sm_flags') >= 10 and 'cf_wide_qk_cache_max' in drv and '_noalpha(' in drv
    src = (R / 'results/cacheflex_flash_attention/bottleneck_20261004/guest_v1/driver.cpp').read_text()
    i = src.index('                PHASE(softmax);\n') + len('                PHASE(softmax);\n'); k = src.index('                PHASE(pv);\n', i)
    assert src[i:k] in drv, 'flag-0 softmax path is not the verbatim FA-39 code'
    print('guest frozen hashes ok; flag-0 softmax path is verbatim FA-39')


if __name__ == '__main__':
    test_plan(); test_command_matches_fa39(); test_config_negative(); test_guest(); print('ALL HOST TESTS PASS')
